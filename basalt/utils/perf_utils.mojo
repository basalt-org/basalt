from time import now
from math import min
from memory import memset, memcpy

from basalt.autograd.node import Node


fn fit_string[num: Int](s: String) -> String:
    var data = DTypePointer[DType.int8]().alloc(num + 1)

    # Copy the the string up to the length of the buffer
    # Fill the rest with spaces & Terminate with zero byte
    memcpy(data, s._as_ptr(), min(num, len(s)))
    if num - min(num, len(s)) > 0:
        memset(data + min(num, len(s)), ord(" "), num - min(num, len(s)))
    data[num] = 0

    return String(data, num + 1)


fn truncate_decimals[num: Int](s: String) -> String:
    var truncated: String
    try:
        var p1 = s.split(delimiter=".")
        truncated = p1[0]
        if len(p1) > 1:
            var p2 = p1[1].split(delimiter="e")
            truncated += "." + fit_string[num](p2[0])
            if len(p2) > 1:
                truncated += "e" + p2[1]

    except e:
        print("[WARNING] could not truncate decimals: ", e)
        truncated = s
    return truncated


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
    ](self, time_format: String = "ns", print_shape: Bool = False):
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

        # 1. Header
        var header = fit_string[5]("Node") + "| " + fit_string[15](
            "Operator"
        ) + "| " + fit_string[20]("Time [" + time_format + "]") + "| " + fit_string[20](
            "Percentage [%]"
        )
        if print_shape:
            header += "| " + fit_string[70]("Shape\t <out> = OP( <in1>, <in2>, <in3> )")
        print(header)

        # 2. Seperator
        var sep = DTypePointer[DType.int8]().alloc(len(header) + 1)
        memset(sep, ord("-"), len(header))
        sep[len(header)] = 0
        var seperator = String(sep, len(header) + 1)
        print(seperator)

        # 3. Perf Data
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

            var print_value = fit_string[5](str(i)) + "| " + fit_string[15](
                value.node.operator
            ) + "| " + fit_string[20](
                truncate_decimals[4](time_converted)
            ) + "| " + fit_string[
                20
            ](
                truncate_decimals[3]((time / total_time) * 100) + " %"
            ) + "| "

            if print_shape:
                var shape_str: String = ""
                shape_str += fit_string[15]("<" + str(value.node.output.shape) + ">")
                shape_str += fit_string[7](" = OP(")
                shape_str += fit_string[15]("<" + str(value.node.input_1.shape) + ">")
                if value.node.input_2:
                    shape_str += ", " + fit_string[15](
                        "<" + str(value.node.input_2.value().shape) + ">"
                    )
                if value.node.input_3:
                    shape_str += ", " + fit_string[15](
                        "<" + str(value.node.input_3.value().shape) + ">"
                    )
                shape_str += ")"

                print_value += shape_str

            print(print_value)

        var total_time_converted = total_time
        if time_format == "ms":
            total_time_converted = total_time / 1e6
        elif time_format == "s":
            total_time_converted = total_time / 1e9
        print(
            "\nTotal average "
            + type_part
            + " time: "
            + str(total_time_converted)
            + " "
            + time_format
        )

    fn print_forward_perf_metrics(
        self, time_format: String = "ns", print_shape: Bool = False
    ):
        self.print_perf_metrics["Forward"](time_format, print_shape)

    fn print_backward_perf_metrics(
        self, time_format: String = "ns", print_shape: Bool = False
    ):
        self.print_perf_metrics["Backward"](time_format, print_shape)
