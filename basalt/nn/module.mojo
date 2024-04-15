from collections.optional import Optional

from sys import env_get_int

from basalt import dtype, TENSORS, GRADS
from basalt import Graph, Symbol, Tensor, TensorShape
from basalt.autograd.ops import forward_op, backward_op
from basalt.utils.tensorutils import fill
from basalt.utils.perf_utils import PerfMetrics
from basalt.utils.collection import Collection
from .initializers import initialize_tensor


# When runing mojo -D DEBUG=1 -I . file, a crash happens at some point at runtime because of an error in linking it seems (because of using -I .)
# For now it seems one has to change this variable manually to be able to run model with performance metrics.
alias DEBUG = env_get_int["DEBUG", 0]()


struct Module[g: Graph]():
    var perf_metrics: PerfMetrics

    fn __init__(inout self):
        @parameter
        if DEBUG == 1:
            self.perf_metrics = PerfMetrics(g)
        else:
            self.perf_metrics = PerfMetrics()

        self.initialize_tensor_memory()
        self.initialize_grad_memory()


    fn forward(inout self, *t_inputs: Tensor[dtype]) -> List[Tensor[dtype]]:
        return self.forward(t_inputs ^)


    fn forward(inout self, t_inputs: VariadicListMem[Tensor[dtype]]) -> List[Tensor[dtype]]:
        # NOTE: The order of the inputs must be the same as the order the inputs were defined in the graph.
        # Example: If you were te define the y_true before the x when creating the graph
        #
        #   var g = Graph()
        #   var y_true = g.input(TensorShape(batch_size, n_outputs))
        #   var x = g.input(TensorShape(batch_size, n_inputs))
        #
        # Then the order of the inputs in the forward call must be the same:
        #
        #   model.forward(batch.labels, batch.inputs)

        # 1. Write inputs to allocated input memory
        for i in range(len(g.inputs)):
            TENSORS[g.inputs[i]] = t_inputs[i]

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
                    TENSORS[out], TENSORS[t1]
                )
            elif op.num_operands == 2:
                # Binary operator
                alias t2 = g.nodes[i].input_2.value()
                forward_op[op, t1.shape, t2.shape, attrs](
                    TENSORS[out],
                    TENSORS[t1],
                    TENSORS[t2],
                )
            elif op.num_operands == 3:
                # Ternary operator
                alias t2 = g.nodes[i].input_2.value()
                alias t3 = g.nodes[i].input_3.value()
                forward_op[op, t1.shape, t2.shape, t3.shape, attrs](
                    TENSORS[out],
                    TENSORS[t1],
                    TENSORS[t2],
                    TENSORS[t3],
                )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_forward_pass(i)

        unroll[fw_unroll, g.nodes.size]()

        # 3. Return outputs from allocated output memory
        # TODO: known copies (reference?)
        var outputs = List[Tensor[dtype]]()
        for i in range(len(g.outputs)):
            outputs.append(TENSORS[g.outputs[i]])
        
        return outputs ^


    fn backward(inout self):

        # 1. Loop over all nodes in reverse order and execute backward operations
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
                        GRADS[out],
                        TENSORS[t1],
                        GRADS[t1],  # grad to be updated: input_1
                    )

            elif op.num_operands == 2:
                # Binary operator
                alias t2 = g.nodes[reverse_i].input_2.value()

                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, t2.shape, attrs](
                        GRADS[out],
                        TENSORS[t1],
                        TENSORS[t2],
                        GRADS[t1],  # grad to be updated: input_1
                    )

                @parameter
                if t2.trainable:
                    backward_op[1, op, out.shape, t1.shape, t2.shape, attrs](
                        GRADS[out],
                        TENSORS[t1],
                        TENSORS[t2],
                        GRADS[t2],  # grad to be updated: input_2
                    )

            elif op.num_operands == 3:
                # Ternary operator
                alias t2 = g.nodes[reverse_i].input_2.value()
                alias t3 = g.nodes[reverse_i].input_3.value()

                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        GRADS[out],
                        TENSORS[t1],
                        TENSORS[t2],
                        TENSORS[t3],
                        GRADS[t1],  # grad to be updated: input_1
                    )

                @parameter
                if t2.trainable:
                    backward_op[1, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        GRADS[out],
                        TENSORS[t1],
                        TENSORS[t2],
                        TENSORS[t3],
                        GRADS[t2],  # grad to be updated: input_2
                    )

                @parameter
                if t3.trainable:
                    backward_op[2, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        GRADS[out],
                        TENSORS[t1],
                        TENSORS[t2],
                        TENSORS[t3],
                        GRADS[t3],  # grad to be updated: input_3
                    )

            # Save end time for performance metrics
            @parameter
            if DEBUG == 1:
                self.perf_metrics.end_backward_pass(i)

        unroll[bw_unroll, g.nodes.size]()

    fn initialize_tensor_memory(inout self):
        for i in range(len(g.inputs)):
            TENSORS.append(Tensor[dtype](g.inputs[i].shape), g.inputs[i])

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

            TENSORS.append(par ^, p)

        for i in range(len(g.nodes)):
            # Assumption: There is only one output tensor per node
            # Assumption: An input or a param cannot be an output of a node
            TENSORS.append(
                Tensor[dtype](g.nodes[i].output.shape), g.nodes[i].output
            )

    fn initialize_grad_memory(inout self):
        # Inputs don't have gradients.
        # Gradient have same shape as the tensor
        for i in range(len(g.params)):
            var grad = g.params.symbols[i]
            if grad.trainable:
                GRADS.append(Tensor[dtype](grad.shape), grad)

        for i in range(len(g.nodes)):
            var out = g.nodes[i].output
            if out.trainable:
                GRADS.append(Tensor[dtype](out.shape), out)

    fn print_perf_metrics(self, time_format: String = "ns", print_shape: Bool = False):
        self.perf_metrics.print_forward_perf_metrics(time_format, print_shape)
        self.perf_metrics.print_backward_perf_metrics(time_format, print_shape)