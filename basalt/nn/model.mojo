from collections.optional import Optional

from sys import env_get_int
from time import now

from basalt.autograd.node import Node
from basalt import Graph, Symbol, Tensor, TensorShape
from basalt.autograd.ops import forward_op, backward_op
from basalt.utils.collection import Collection
from basalt.utils.tensorutils import fill
from .initializers import initialize_tensor


# When runing mojo -D DEBUG=1 -I . file, a crash happens at some point at runtime because of an error in linking it seems (because of using -I .) 
# For now it seems one has to change this variable manually to be able to run model with performance metrics.
alias DEBUG = env_get_int["DEBUG", 0]()


@value
struct PerfMetricsValues(CollectionElement):
    var node: Node
    var time: Float64

    fn __init__(inout self, node: Node, time: Float64):
        self.node = node
        self.time = time


struct PerfMetrics:
    # values are in "ns"
    # using perf_metrics can reduce the speed of each epoch of the model a little bit
    var forward_perf_metrics: List[PerfMetricsValues]
    var backward_perf_metrics: List[PerfMetricsValues]
    var epochs_forward: Int
    var epochs_backward: Int
    var start: Int


    fn __init__(inout self):
        self.forward_perf_metrics = List[PerfMetricsValues]()
        self.backward_perf_metrics = List[PerfMetricsValues]()
        self.epochs_forward = 0
        self.epochs_backward = 0
        self.start = 0

    fn __init__(inout self, graph: Graph):
        self.forward_perf_metrics = List[PerfMetricsValues]()
        self.backward_perf_metrics = List[PerfMetricsValues]()

        for i in range(graph.nodes.size):
            self.forward_perf_metrics.append(PerfMetricsValues(graph.nodes[i], 0.0))
            self.backward_perf_metrics.append(PerfMetricsValues(graph.nodes[i], 0.0))

        self.epochs_forward = 0
        self.epochs_backward = 0
        self.start = 0

    fn start_forward_pass(inout self):
        self.start = now()

    fn end_forward_pass(inout self, pos: Int):
        # Change this to use references when list has them available
        var old_value = self.forward_perf_metrics[pos]
        self.forward_perf_metrics[pos] = PerfMetricsValues(old_value.node, old_value.time + (now() - self.start))
        self.epochs_forward += 1

    fn start_backward_pass(inout self):
        self.start = now()
    
    fn end_backward_pass(inout self, pos: Int):
        var old_value = self.backward_perf_metrics[pos]
        self.backward_perf_metrics[pos] = PerfMetricsValues(old_value.node, old_value.time + (now() - self.start))
        self.epochs_backward += 1

    fn print_perf_metrics[type_part: String](inout self, time_format: String = "ns", print_shape: Bool = False):
        if type_part == "Forward" and len(self.forward_perf_metrics) == 0:
            return
        if type_part == "Backward" and len(self.backward_perf_metrics) == 0:
            return

        if type_part == "Forward":
            print("\n\nForward pass performance metrics:")
        else:
            print("\n\nBackward pass performance metrics:")

        var total_time: SIMD[DType.float64, 1] = 0

        var size: Int = 0
        @parameter
        if type_part == "Forward":
            size = len(self.forward_perf_metrics)
        elif type_part == "Backward":
            size = len(self.backward_perf_metrics)
        for i in range(size):
            @parameter
            if type_part == "Forward":
                total_time += self.forward_perf_metrics[i].time
            elif type_part == "Backward":
                total_time += self.backward_perf_metrics[i].time
    
        for i in range(len(self.forward_perf_metrics)):
            var value: PerfMetricsValues
            @parameter
            if type_part == "Forward":
                value = self.forward_perf_metrics[i]
            else:
                value = self.backward_perf_metrics[i]

            var time = value.time / self.epochs_forward
            if time_format == "ms":
                time = time / 1e6
            elif time_format == "s":
                time = time / 1e9

            var print_value = "Node: " + str(i) + " Operator: " + value.node.operator + " Time: " + time + time_format + " Percentage of time taken: " + (value.time / total_time) * 100 + "%. "
            if print_shape:
                print_value += "Input shape 1: " + str(value.node.input_1.shape)
                if value.node.input_2:
                    print_value += " Input shape 2: " + str(value.node.input_2.value().shape)
                if value.node.input_3:
                    print_value += " Input shape 3: " + str(value.node.input_3.value().shape)
                print_value += " Output shape: " + str(value.node.output.shape)
            print(print_value)   

    fn print_forward_perf_metrics(inout self, time_format: String = "ns", print_shape: Bool = False):
        self.print_perf_metrics["Forward"](time_format, print_shape)

    fn print_backward_perf_metrics(inout self, time_format: String = "ns", print_shape: Bool = False):
        self.print_perf_metrics["Backward"](time_format, print_shape)


fn dv_contains(dv: List[Symbol], symbol: Symbol) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False


# TODO: remove when ability to concatenate graphs
fn calc_n_inference_nodes(g: Graph) -> Optional[Int]:
    """
    Calculate the index of the node up to wich the forward pass should be executed for a model inference.
    When looping in revers: Equals the first index on which the node output is also a graph output.
    The number of inference nodes is that index + 1.
    """
    for i in range(len(g.nodes) - 1, -1, -1):
        if dv_contains(g.outputs, g.nodes[i].output):
            return i + 1
    return None


fn collect_trainable_parameters(g: Graph) -> List[Symbol]:
    """
    Collect all symbols of trainable parameters.
    """

    var trainable_parameters = List[Symbol]()

    for i in range(len(g.params)):
        if g.params.symbols[i].trainable:
            trainable_parameters.append(g.params.symbols[i])

    return trainable_parameters ^


struct Parameters[g: Graph]():
    var params: Collection
    var grads: Collection

    alias trainable_parameters = collect_trainable_parameters(g)

    fn __init__(inout self):
        # Max number of tensors to initialize (max capacity to avoid resizing)
        # Assumption: An input or a param cannot be an output of a node
        # Assumption: There is only one output tensor per node
        var N = len(g.inputs) + len(g.params) + len(g.nodes)
        self.params = Collection(capacity=N)
        self.grads = Collection(capacity=N)


struct Model[
    g: Graph,
    n_inference_nodes: Optional[Int] = calc_n_inference_nodes(g),  # TODO: remove this
]():
    var parameters: Parameters[g]
    var perf_metrics: PerfMetrics

    fn __init__(inout self, inference_only: Bool = False):
        @parameter
        if DEBUG == 1:
            self.perf_metrics = PerfMetrics(g)
        else:
            self.perf_metrics = PerfMetrics()

        self.parameters = Parameters[g]()
        self.allocate_tensor_memory()
        self.allocate_grad_memory()

        # TODO: ability to concatenate graphs
        # NOTE: inference_only only used for surpressing the warning.
        if not inference_only and not g.loss_out:
            print("\n\n[WARNING]: No loss defined, model.forward() unavailable!\n\n")
        if not n_inference_nodes:
            print(
                "\n\n[WARNING]: No graph out defined, model.inference()"
                " unavailable!\n\n"
            )

    # TODO: ability to concatenate graphs
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
        return self.parameters.params[g.loss_out.value()]

    fn inference(inout self, *t_inputs: Tensor[dtype]) -> List[Tensor[dtype]]:
        # 1. Execute forward pass up to model out
        self.execute[n_inference_nodes.value()](t_inputs)

        # 2. Return outputs from allocated output memory
        # TODO: known copies (reference?)
        var outputs = List[Tensor[dtype]]()
        for i in range(len(g.outputs)):
            outputs.append(self.parameters.params[g.outputs[i]])
        return outputs ^

    fn execute[num_nodes: Int](inout self, t_input: VariadicListMem[Tensor[dtype]]):
        # 1. Write inputs to allocated input memory
        for i in range(len(g.inputs)):
            self.parameters.params[g.inputs[i]] = t_input[i]

        # 2. Loop over all nodes and execute forward operations
        @parameter
        fn fw_unroll[i: Int]():
            alias op = g.nodes[i].operator
            alias t1 = g.nodes[i].input_1
            alias out = g.nodes[i].output
            alias attrs = g.nodes[i].attributes

            # Save start time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.start_forward_pass()

            @parameter
            if op.num_operands == 1:
                # Unary operator
                forward_op[op, t1.shape, attrs](
                    self.parameters.params[out], self.parameters.params[t1]
                )
            elif op.num_operands == 2:
                # Binary operator
                alias t2 = g.nodes[i].input_2.value()
                forward_op[op, t1.shape, t2.shape, attrs](
                    self.parameters.params[out],
                    self.parameters.params[t1],
                    self.parameters.params[t2],
                )
            elif op.num_operands == 3:
                # Ternary operator
                alias t2 = g.nodes[i].input_2.value()
                alias t3 = g.nodes[i].input_3.value()
                forward_op[op, t1.shape, t2.shape, t3.shape, attrs](
                    self.parameters.params[out],
                    self.parameters.params[t1],
                    self.parameters.params[t2],
                    self.parameters.params[t3],
                )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_forward_pass(i)

        unroll[fw_unroll, num_nodes]()

    fn backward(inout self):
        """
        Main entrypoint of backward pass.
        """

        # 1. Initialize output gradient at the beginning of the backward pass
        fill(self.parameters.grads[g.loss_out.value()], 1.0)

        # 2. Loop over all nodes in reverse order and execute backward operations
        @parameter
        fn bw_unroll[i: Int]():
            alias reverse_i = g.nodes.size - i - 1
            alias op = g.nodes[reverse_i].operator
            alias out = g.nodes[reverse_i].output  # or upper_grad symbol
            alias t1 = g.nodes[reverse_i].input_1
            alias attrs = g.nodes[reverse_i].attributes

            # Save start time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.start_backward_pass()

            @parameter
            if op.num_operands == 1:
                # Unary operator
                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.grads[t1],  # grad to be updated: input_1
                    )

            elif op.num_operands == 2:
                # Binary operator
                alias t2 = g.nodes[reverse_i].input_2.value()

                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, t2.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.grads[t1],  # grad to be updated: input_1
                    )

                @parameter
                if t2.trainable:
                    backward_op[1, op, out.shape, t1.shape, t2.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.grads[t2],  # grad to be updated: input_2
                    )

            elif op.num_operands == 3:
                # Ternary operator
                alias t2 = g.nodes[reverse_i].input_2.value()
                alias t3 = g.nodes[reverse_i].input_3.value()

                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.params[t3],
                        self.parameters.grads[t1],  # grad to be updated: input_1
                    )

                @parameter
                if t2.trainable:
                    backward_op[1, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.params[t3],
                        self.parameters.grads[t2],  # grad to be updated: input_2
                    )

                @parameter
                if t3.trainable:
                    backward_op[2, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.params[t3],
                        self.parameters.grads[t3],  # grad to be updated: input_3
                    )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_backward_pass(i)

        unroll[bw_unroll, g.nodes.size]()

    fn allocate_tensor_memory(inout self):
        for i in range(len(g.inputs)):
            self.parameters.params.append(Tensor[dtype](g.inputs[i].shape), g.inputs[i])

        for i in range(len(g.params)):
            var p = g.params.symbols[i]
            var p_init = g.params.values[i]

            var par: Tensor[dtype]
            if p_init.initializer:
                # 1. Specific parameter initialization defined
                var initializer_attr = p_init.initializer.value()
                par = initialize_tensor(
                    shape=p.shape,
                    type=initializer_attr.value.to_string(),
                    data=p_init.data.value(),
                )
            elif p_init.data:
                # 2. Parameter initialized with data only
                # Data is assumed to contain the tensor
                par = g.params.get_tensor(i)
            else:
                # Default parameter initialization to zero
                par = Tensor[dtype](p.shape)

            self.parameters.params.append(par ^, p)

        for i in range(len(g.nodes)):
            # Assumption: There is only one output tensor per node
            # Assumption: An input or a param cannot be an output of a node
            self.parameters.params.append(
                Tensor[dtype](g.nodes[i].output.shape), g.nodes[i].output
            )

    fn allocate_grad_memory(inout self):
        # Inputs don't have gradients.
        # Gradient have same shape as the tensor
        for i in range(len(g.params)):
            var grad = g.params.symbols[i]
            if grad.trainable:
                self.parameters.grads.append(Tensor[dtype](grad.shape), grad)

        for i in range(len(g.nodes)):
            var out = g.nodes[i].output
            if out.trainable:
                self.parameters.grads.append(Tensor[dtype](out.shape), out)
