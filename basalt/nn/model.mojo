from collections.optional import Optional

from sys import env_get_int

from basalt import Graph, Symbol, Tensor, TensorShape
from basalt.autograd.ops import forward_op, backward_op
from basalt.utils.collection import Collection
from basalt.utils.tensorutils import fill
from .initializers import initialize_tensor
from basalt.utils.perf_utils import PerfMetrics


# When runing mojo -D DEBUG=1 -I . file, a crash happens at some point at runtime because of an error in linking it seems (because of using -I .)
# For now it seems one has to change this variable manually to be able to run model with performance metrics.
alias DEBUG = env_get_int["DEBUG", 0]()


# TODO: remove when ability to concatenate graphs (modules)
fn dv_contains(dv: List[Symbol], symbol: Symbol) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False


# TODO: remove when ability to concatenate graphs (modules)
fn n_inference_nodes(g: Graph) -> Optional[Int]:
    """
    Calculate the index of the node up to wich the forward pass should be executed for a model inference.
    When looping in revers: Equals the first index on which the node output is also a graph output.
    The number of inference nodes is that index + 1.
    """
    for i in range(len(g.nodes) - 1, -1, -1):
        for j in range(len(g.nodes[i].outputs)):
            if dv_contains(g.outputs, g.nodes[i].outputs[j]):
                return i + 1
    return None


@value
struct Parameters:
    var tensors: Collection
    var grads: Collection

    fn __init__(inout self):
        self.tensors = Collection()
        self.grads = Collection()


struct Model[
    g: Graph,
    n_inference_nodes: Optional[Int] = n_inference_nodes(g),
]():
    var parameters: Parameters
    var perf_metrics: PerfMetrics

    fn __init__(inout self, inference_only: Bool = False):
        self.parameters = Parameters()

        @parameter
        if DEBUG == 1:
            self.perf_metrics = PerfMetrics(g)
        else:
            self.perf_metrics = PerfMetrics()

        self.allocate_tensor_memory()
        self.allocate_grad_memory()

        # TODO: remove this when ability to concatenate graphs (modules)
        # NOTE: inference_only only used for surpressing the warning.
        if not inference_only and not g.loss_out:
            print("\n\n[WARNING]: No loss defined, model.forward() unavailable!\n\n")
        if not n_inference_nodes:
            print(
                "\n\n[WARNING]: No graph out defined, model.inference()"
                " unavailable!\n\n"
            )

    # TODO: remove when ability to concatenate graphs (modules)
    # Removes the need for splitting in forward and inference mode
    fn forward(inout self, *t_inputs: Tensor[dtype]) -> Tensor[dtype]:
        # NOTE: Important detail here is that the order of the inputs must be the same as the order the inputs were defined in the graph.
        # Example: If you were te define the y_true before the x when creating the graph
        #
        #   var g = Graph()
        #   var y_true = g.input(TensorShape(batch_size, n_outputs))
        #   var x = g.input(TensorShape(batch_size, n_inputs))
        #
        # Then the order of the inputs in the forward call must be the same:
        #
        #   model.forward(batch.labels, batch.inputs)

        # 1. Execute a full forward pass (model inference + loss)
        self.execute[g.nodes.size](t_inputs ^)

        # 2. Return loss from allocated output memory
        # TODO: known copy (reference?)
        return self.parameters.tensors[g.loss_out.value()]

    fn inference(inout self, *t_inputs: Tensor[dtype]) -> List[Tensor[dtype]]:
        # 1. Execute forward pass up to model out
        self.execute[n_inference_nodes.value()](t_inputs)

        # 2. Return outputs from allocated output memory
        # TODO: known copies (reference?)
        var outputs = List[Tensor[dtype]]()
        for i in range(len(g.outputs)):
            outputs.append(self.parameters.tensors[g.outputs[i]])
        return outputs ^

    fn execute[num_nodes: Int](inout self, t_input: VariadicListMem[Tensor[dtype]]):
        # 1. Write inputs to allocated input memory
        for i in range(len(g.inputs)):
            self.parameters.tensors[g.inputs[i]] = t_input[i]

        # 2. Loop over all nodes and execute forward operations
        @parameter
        fn fw_unroll[i: Int]():
            alias op = g.nodes[i].operator
            alias attrs = g.nodes[i].attributes

            # Save start time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.start_forward_pass()

            @parameter
            if op.dynamic:
                forward_op[op, attrs](
                    g.nodes[i].inputs,
                    g.nodes[i].outputs,
                    Reference(self.parameters),
                )
            else:
                # Statically known shapes and number of operands
                alias num_operands = len(g.nodes[i].inputs)
                alias t1 = g.nodes[i].inputs[0]
                alias out = g.nodes[i].outputs[0]

                @parameter
                if num_operands == 1:
                    # Unary operator
                    forward_op[op, t1.shape, attrs](
                        self.parameters.tensors[out], self.parameters.tensors[t1]
                    )
                elif num_operands == 2:
                    # Binary operator
                    alias t2 = g.nodes[i].inputs[1]
                    forward_op[op, t1.shape, t2.shape, attrs](
                        self.parameters.tensors[out],
                        self.parameters.tensors[t1],
                        self.parameters.tensors[t2],
                    )
                elif num_operands == 3:
                    # Ternary operator
                    alias t2 = g.nodes[i].inputs[1]
                    alias t3 = g.nodes[i].inputs[2]
                    forward_op[op, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.tensors[out],
                        self.parameters.tensors[t1],
                        self.parameters.tensors[t2],
                        self.parameters.tensors[t3],
                    )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_forward_pass(i)

        unroll[fw_unroll, num_nodes]()

    fn backward(inout self, *upper_grads: Tensor[dtype]):
        """
        Main entrypoint of backward pass.
        """
        # 1. Initialize output gradient at the beginning of the backward pass
        if len(upper_grads) == 0:
            # TODO remove loss_out tag
            fill(self.parameters.grads[g.loss_out.value()], 1.0)
        else:
            var node_outputs = g.nodes[g.nodes.size - 1].outputs
            if len(upper_grads) != node_outputs.size:
                print(
                    "[WARNING] Number of upper grads does not match number of node"
                    " outputs!"
                )
            for i in range(node_outputs.size):
                self.parameters.grads[node_outputs[i]] = upper_grads[i]

        # 2. Loop over all nodes in reverse order and execute backward operations
        @parameter
        fn bw_unroll[i: Int]():
            alias reverse_i = g.nodes.size - i - 1
            alias op = g.nodes[reverse_i].operator
            alias attrs = g.nodes[reverse_i].attributes
            alias num_operands = len(g.nodes[reverse_i].inputs)

            # Save start time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.start_backward_pass()

            @parameter
            if op.dynamic:

                @parameter
                fn unroll_dynamic[j: Int]():
                    @parameter
                    if g.nodes[reverse_i].inputs[j].trainable:
                        backward_op[j, op, attrs](
                            g.nodes[reverse_i].inputs,
                            g.nodes[reverse_i].outputs,
                            self.parameters.grads[g.nodes[reverse_i].inputs[j]],
                            Reference(self.parameters),
                        )

                unroll[unroll_dynamic, num_operands]()

            else:
                # Statically known shapes and number of operands
                alias out = g.nodes[reverse_i].outputs[0]  # or upper_grad symbol
                alias t1 = g.nodes[reverse_i].inputs[0]

                @parameter
                if num_operands == 1:
                    # Unary operator
                    @parameter
                    if t1.trainable:
                        backward_op[0, op, out.shape, t1.shape, attrs](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.grads[t1],  # grad to be updated: inputs[0]
                        )

                elif num_operands == 2:
                    # Binary operator
                    alias t2 = g.nodes[reverse_i].inputs[1]

                    @parameter
                    if t1.trainable:
                        backward_op[0, op, out.shape, t1.shape, t2.shape, attrs](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.grads[t1],  # grad to be updated: inputs[0]
                        )

                    @parameter
                    if t2.trainable:
                        backward_op[1, op, out.shape, t1.shape, t2.shape, attrs](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.grads[t2],  # grad to be updated: inputs[1]
                        )

                elif num_operands == 3:
                    # Ternary operator
                    alias t2 = g.nodes[reverse_i].inputs[1]
                    alias t3 = g.nodes[reverse_i].inputs[2]

                    @parameter
                    if t1.trainable:
                        backward_op[
                            0, op, out.shape, t1.shape, t2.shape, t3.shape, attrs
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.tensors[t3],
                            self.parameters.grads[t1],  # grad to be updated: inputs[0]
                        )

                    @parameter
                    if t2.trainable:
                        backward_op[
                            1, op, out.shape, t1.shape, t2.shape, t3.shape, attrs
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.tensors[t3],
                            self.parameters.grads[t2],  # grad to be updated: inputs[1]
                        )

                    @parameter
                    if t3.trainable:
                        backward_op[
                            2, op, out.shape, t1.shape, t2.shape, t3.shape, attrs
                        ](
                            self.parameters.grads[out],
                            self.parameters.tensors[t1],
                            self.parameters.tensors[t2],
                            self.parameters.tensors[t3],
                            self.parameters.grads[t3],  # grad to be updated: inputs[2]
                        )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_backward_pass(i)

        unroll[bw_unroll, g.nodes.size]()

    fn allocate_tensor_memory(inout self):
        for i in range(len(g.inputs)):
            self.parameters.tensors.append(
                Tensor[dtype](g.inputs[i].shape), g.inputs[i]
            )

        for i in range(len(g.params)):
            var p = g.params.symbols[i]
            var p_init = g.params.values[i]

            var par: Tensor[dtype]
            if p_init.initializer:
                # 1. Specific parameter initialization defined
                var initializer_attr = p_init.initializer.value()
                par = initialize_tensor(
                    shape=p.shape,
                    type=initializer_attr.to_string(),
                    data=p_init.data.value(),
                )
            elif p_init.data:
                # 2. Parameter initialized with data only
                # Data is assumed to contain the tensor
                par = g.params.get_tensor(i)
            else:
                # Default parameter initialization to zero
                par = Tensor[dtype](p.shape)

            self.parameters.tensors.append(par ^, p)

        for i in range(len(g.nodes)):
            # Assumption: An input or a param cannot be an output of a node
            for j in range(len(g.nodes[i].outputs)):
                self.parameters.tensors.append(
                    Tensor[dtype](g.nodes[i].outputs[j].shape), g.nodes[i].outputs[j]
                )

    fn allocate_grad_memory(inout self):
        # Gradient have same shape as the tensor
        for i in range(len(g.inputs)):
            if g.inputs[i].trainable:
                self.parameters.grads.append(
                    Tensor[dtype](g.inputs[i].shape), g.inputs[i]
                )

        for i in range(len(g.params)):
            var grad = g.params.symbols[i]
            if grad.trainable:
                self.parameters.grads.append(Tensor[dtype](grad.shape), grad)

        for i in range(len(g.nodes)):
            for j in range(len(g.nodes[i].outputs)):
                var out = g.nodes[i].outputs[j]
                if out.trainable:
                    self.parameters.grads.append(Tensor[dtype](out.shape), out)

    fn print_perf_metrics(self, time_format: String = "ns", print_shape: Bool = False):
        self.perf_metrics.print_forward_perf_metrics(time_format, print_shape)
        self.perf_metrics.print_backward_perf_metrics(time_format, print_shape)
