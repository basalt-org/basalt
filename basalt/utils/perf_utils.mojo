from time import now

from basalt.autograd.node import Node

@value
struct PerfMetricsValues(CollectionElement):
    var node: Node
    var time: Float64

    fn __init__(inout self, node: Node, time: Float64):
        self.node = node
        self.time = time


@value
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
        self.forward_perf_metrics[pos] = PerfMetricsValues(
            old_value.node, old_value.time + (now() - self.start)
        )
        self.epochs_forward += 1

    fn start_backward_pass(inout self):
        self.start = now()

    fn end_backward_pass(inout self, pos: Int):
        var old_value = self.backward_perf_metrics[pos]
        self.backward_perf_metrics[pos] = PerfMetricsValues(
            old_value.node, old_value.time + (now() - self.start)
        )
        self.epochs_backward += 1

    fn print_perf_metrics[
        type_part: String
    ](inout self, time_format: String = "ns", print_shape: Bool = False):
        # Calculates the average time for each node operation.

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
                total_time += self.forward_perf_metrics[i].time / self.epochs_forward
            elif type_part == "Backward":
                total_time += self.backward_perf_metrics[i].time / self.epochs_backward

        for i in range(len(self.forward_perf_metrics)):
            var value: PerfMetricsValues

            @parameter
            if type_part == "Forward":
                value = self.forward_perf_metrics[i]
            else:
                value = self.backward_perf_metrics[i]

            var time = value.time

            @parameter
            if type_part == "Forward":
                time = time / self.epochs_forward
            else:
                time = time / self.epochs_backward

            var time_converted = time
            if time_format == "ms":
                time_converted = time / 1e6
            elif time_format == "s":
                time_converted = time / 1e9

            var print_value = "Node: " + str(i) 
                + " Operator: " + value.node.operator + " Time: " 
                + time_converted + time_format + " Percentage of time taken: " 
                + (time / total_time) * 100 + "%. "

            if print_shape:
                print_value += "Input shape 1: " + str(value.node.input_1.shape)
                if value.node.input_2:
                    print_value += " Input shape 2: " + str(
                        value.node.input_2.value().shape
                    )
                if value.node.input_3:
                    print_value += " Input shape 3: " + str(
                        value.node.input_3.value().shape
                    )
                print_value += " Output shape: " + str(value.node.output.shape)
            print(print_value)

    fn print_forward_perf_metrics(
        inout self, time_format: String = "ns", print_shape: Bool = False
    ):
        self.print_perf_metrics["Forward"](time_format, print_shape)

    fn print_backward_perf_metrics(
        inout self, time_format: String = "ns", print_shape: Bool = False
    ):
        self.print_perf_metrics["Backward"](time_format, print_shape)

