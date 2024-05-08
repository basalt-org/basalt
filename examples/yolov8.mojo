import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP, dtype
# TODO: I dont love this style of imports, I will probably rework it to be more explicit later on
from math import ceil


fn Conv(
    inout g: Graph,
    x: Symbol,
    out_channels: Int,
    kernel_size: Int,
    padding: Int,
    stride: Int,
) -> Symbol:
    # NOTE: This is functionally equivalent to the Conv2D -> BatchNorm2D (removed in graph) -> SiLU (According to ONNX)
    var conv = nn.Conv2d(g, x, out_channels, kernel_size, padding, stride)
    var sigmoid = g.op(OP.SIGMOID, conv)
    return g.op(OP.MUL, conv, sigmoid)


fn C2f(
    inout g: Graph,
    x: Symbol,
    out_channels: Int,
    n: Int,
    shortcut: Bool
) -> Symbol:
    var conv = nn.Conv2d(g, x, out_channels, 1, 0, 1)

    var split_size = out_channels // 2
    var split_sections = List[Int](split_size, split_size)
    var split = g.split(conv, split_sections, dim=1)

    @parameter
    fn bottleneck(
        x: Symbol, out_channels: Int, shortcut: Bool = False
    ) -> Symbol:
        var conv1 = Conv(g, x, split_size, 1, 1, 1)
        var conv2 = Conv(g, conv1, split_size, 3, 1, 1)

        if shortcut:
            return g.op(OP.ADD, x, conv2)
        else:
            return conv2

    var y1 = bottleneck(split[1], split_size, shortcut)
    var y2 = y1

    # NOTE: This assumes n >= 1 (Could add a constrained for it later)
    for i in range(1, n):
        y2 = bottleneck(y2, split_size, shortcut)

    var y = g.concat(split[0], y1, y2, dim=1)

    return Conv(g, y, out_channels, 1, 0, 1)


fn SPPF(inout g: Graph, x: Symbol, out_channels: Int) -> Symbol:
    var conv = Conv(g, x, out_channels, 1, 0, 1)

    var maxpool2d_1 = nn.MaxPool2d(g, x, kernel_size=5, padding=2)
    var maxpool2d_2 = nn.MaxPool2d(g, x, kernel_size=5, padding=2)
    var maxpool2d_3 = nn.MaxPool2d(g, x, kernel_size=5, padding=2)

    var y = g.concat(conv, maxpool2d_1, maxpool2d_2, maxpool2d_3, dim=1)

    return Conv(g, y, out_channels, 1, 0, 1)


fn YoloV8(batch_size: Int, yolo_model_type: StaticTuple[Float64, 3]) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 3, 640, 640))

    # Adapted from https://private-user-images.githubusercontent.com/27466624/239739723-57391d0f-1848-4388-9f30-88c2fb79233f.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTUxMTk0MDYsIm5iZiI6MTcxNTExOTEwNiwicGF0aCI6Ii8yNzQ2NjYyNC8yMzk3Mzk3MjMtNTczOTFkMGYtMTg0OC00Mzg4LTlmMzAtODhjMmZiNzkyMzNmLmpwZz9Y>LUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTA3VDIxNTgyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNlZTdkY2ZiMDA0Y2VlOGZkYjllN2FkYTQ1MTY5OWY1YzYwNjIxZDM4OTZiYWRiMGU5YWQxNzkyMTcwNGNmNTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.0ocPCiokkivvk95bQCds6Nt0EblUrHZElycV311ImF4. Some values (output_channels, stride, etc..) are different in the onnx file and the graph image.

    # Backbone
    var out_channels_1 = int(64 * yolo_model_type[2])
    var conv_1 = Conv(g, x, out_channels_1, 3, 1, 2)
    var out_channels_2 = int(128 * yolo_model_type[1])
    var conv_2 = Conv(g, conv_1, out_channels_2, 3, 1, 2)
    var C2F_n_1 = int(ceil(3 * yolo_model_type[0]))
    var C2f_1 = C2f(g, conv_2, out_channels_2, n=C2F_n_1)
    var out_channels_3 = int(256 * yolo_model_type[1])
    var conv_3 = Conv(g, C2f_1, out_channels_3, 3, 1, 2)
    var C2F_n_2 = int(ceil(6 * yolo_model_type[0]))
    var C2f_2 = C2f(g, conv_3, out_channels_3, n=C2F_n_2)

    var out_channels_4 = int(512 * yolo_model_type[1])
    var conv_4 = Conv(g, C2f_2, out_channels_4, 3, 1, 2)
    var C2f_3 = C2f(g, conv_4, out_channels_4, n=C2F_n_2)

    var out_channels_5 = int(512 * yolo_model_type[1] * yolo_model_type[2])
    var conv_5 = Conv(g, C2f_3, out_channels_5, 3, 1, 2)
    var C2f_4 = C2f(g, conv_5, out_channels_5, n=C2F_n_1)
    var SPPF_1 = SPPF(g, C2f_4, out_channels_5, 32)

    return g


alias yolov8_n = StaticTuple[Float64, 3](
    0.33, 0.25, 2
)  # d (depth_multiplier), w (width_multiplier), r (ratio)
# var yolov8_s
# var yolov8_m


fn main():
    alias graph = YoloV8(4, yolov8_n)
    var model = nn.Model[graph]()
    ...
