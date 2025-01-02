from time.time import monotonic as now
from memory import UnsafePointer, memcpy, memset

from basalt.autograd.node import Node


@always_inline("nodebug")
fn fit_string[num: Int](s: String) -> String:
    var data = UnsafePointer[Byte]().alloc(num + 1)
    var copy_len = min(num, len(s))

    memcpy(data, s.unsafe_ptr(), copy_len)
    memset(data + copy_len, ord(" "), num - copy_len)
    data[num] = 0

    return String(ptr=data, length=num + 1)


@always_inline("nodebug")
fn truncate_decimals[num: Int](s: String) -> String:
    try:
        var parts = s.split(".")
        var truncated = parts[0]

        if len(parts) > 1:
            var decimal_parts = parts[1].split("e")
            truncated += "." + fit_string[num](decimal_parts[0])

            if len(decimal_parts) > 1:
                truncated += "e" + decimal_parts[1]

        return truncated
    except e:
        print("[WARNING] could not truncate decimals: ", e)
        return s


@value
struct PerfMetricsValues:
    var node: Node
    var ns: Float64


@value
struct PerfMetrics:
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

        self.forward_perf_metrics.reserve(graph.nodes.size)
        self.backward_perf_metrics.reserve(graph.nodes.size)

        for i in range(graph.nodes.size):
            self.forward_perf_metrics.append(PerfMetricsValues(graph.nodes[i], 0.0))
            self.backward_perf_metrics.append(PerfMetricsValues(graph.nodes[i], 0.0))

        self.epochs_forward = 0
        self.epochs_backward = 0
        self.start = 0

    fn start_forward_pass(inout self):
        self.start = now()

    fn end_forward_pass(inout self, pos: Int):
        self.forward_perf_metrics[pos].ns += now() - self.start
        self.epochs_forward += 1

    fn start_backward_pass(inout self):
        self.start = now()

    fn end_backward_pass(inout self, pos: Int):
        self.backward_perf_metrics[pos].ns += now() - self.start
        self.epochs_backward += 1

    fn print_perf_metrics[
        type_part: String
    ](self, time_format: String = "ns", print_shape: Bool = False):
        constrained[type_part == "Forward" or type_part == "Backward", "Only 'Forward' or 'Backward' are accepted types."]()
        
        alias is_forward = type_part == "Forward"

        var metrics = self.forward_perf_metrics if is_forward else self.backward_perf_metrics
        var epochs = self.epochs_forward if is_forward else self.epochs_backward
        var size = len(metrics)
        var total_time: Float64 = 0

        if size == 0:
            return

        if is_forward:
            print("\n\nForward pass performance metrics:")
        else:
            print("\n\nBackward pass performance metrics:")
            
        for i in range(size):
            total_time += metrics[i].ns

        var header = (
            fit_string[5]("Node")
            + "| "
            + fit_string[15]("Operator")
            + "| "
            + fit_string[20]("Time [" + time_format + "]")
            + "| "
            + fit_string[20]("Percentage [%]")
        )

        if print_shape:
            header += "| " + fit_string[70]("Shape\t <out> = OP( <in1>, <in2>, <in3> )")
            
        print(header)

        var header_length = len(header)
        var seperator = UnsafePointer[UInt8]().alloc(header_length + 1)
        
        memset(seperator, ord("-"), header_length)
        seperator[header_length] = 0
        
        print(String(ptr=seperator, length=len(header) + 1))

        for i in range(size):
            var value = metrics[i]
            var time = value.ns / epochs

            if time_format == "ms":
                time /=  1e6
            elif time_format == "s":
                time /= 1e9

            var percentage = (value.ns / total_time) * 100

            var print_value = (
                fit_string[5](str(i))
                + "| "
                + fit_string[15](str(value.node.operator))
                + "| "
                + fit_string[20](truncate_decimals[4](str(time)))
                + "| "
                + fit_string[20](truncate_decimals[3](str(percentage)) + " %")
                + "| "
            )

            if print_shape:
                var shape_str = fit_string[15]("<" + str(value.node.outputs[0].shape) + ">")

                for j in range(1, len(value.node.outputs)):
                    shape_str += ", " + fit_string[15]("<" + str(value.node.outputs[j].shape) + ">")

                shape_str += fit_string[7](" = OP(") + fit_string[15]("<" + str(value.node.inputs[0].shape) + ">")

                for j in range(1, len(value.node.inputs)):
                    shape_str += ", " + fit_string[15]("<" + str(value.node.inputs[j].shape) + ">")

                shape_str += ")"

                print(print_value, end="")
                print(shape_str)
            else:
                print(print_value)

        if time_format == "ms":
            total_time /=  1e6
        elif time_format == "s":
            total_time /= 1e9

        print(
            "\nTotal average "
            + type_part
            + " time: "
            + str(total_time)
            + " "
            + time_format
        )


    fn print_forward_perf_metrics(self, time_format: String = "ns", print_shape: Bool = False):
        self.print_perf_metrics["Forward"](time_format, print_shape)

    fn print_backward_perf_metrics(self, time_format: String = "ns", print_shape: Bool = False):
        self.print_perf_metrics["Backward"](time_format, print_shape)
