import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP, dtype
from basalt.autograd.attributes import AttributeVector, Attribute
from basalt.utils.tensor_creation_utils import to_tensor, to_numpy

from python import Python
from math import ceil
from utils.static_tuple import StaticTuple


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


fn Conv(
    inout g: Graph,
    x: Symbol,
    weight: Symbol,
    bias: Symbol,
    kernel_size: StaticIntTuple[2],
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
) -> Symbol:
    # NOTE: This is functionally equivalent to the Conv2D -> BatchNorm2D (removed in graph) -> SiLU (According to ONNX)
    var conv = g.op(OP.CONV2D, x, weight, bias, attributes=AttributeVector(
        Attribute("padding", padding),
        Attribute("stride", stride),
        Attribute("dilation", StaticIntTuple[2](1, 1)),
    ))
    var sigmoid = g.op(OP.SIGMOID, conv)
    return g.op(OP.MUL, conv, sigmoid)


fn C2f(
    inout g: Graph,
    x: Symbol,
    out_channels: Int,
    n: Int,
    shortcut: Bool
) -> Symbol:
    var conv = Conv(g, x, out_channels, 1, 0, 1)

    var split_size = out_channels // 2
    var split_sections = List[Int](split_size, split_size)
    var split = g.split(conv, split_sections, dim=1)

    # declare the weights for the last conv here because that is the order in onnx file
    var n_temp = 1
    if n > 1:
        n_temp = 2
    var weight = g.param(TensorShape(out_channels, split_size * (n + 2), 1, 1))
    var bias = g.param(TensorShape(out_channels))

    @parameter
    fn bottleneck(
        x: Symbol, out_channels: Int, shortcut: Bool = False
    ) -> Symbol:
        var conv1 = Conv(g, x, out_channels, 3, 1, 1)
        var conv2 = Conv(g, conv1, out_channels, 3, 1, 1)

        if shortcut:
            return g.op(OP.ADD, x, conv2)
        else:
            return conv2

    var y1 = bottleneck(split[1], split_size, shortcut)
    var y2 = y1

    var concat_list = List[Symbol]() # add ability to concat to receive a list, becauase the the concatenation has to be done for each bottleneck layer that was run

    # NOTE: This assumes n >= 1 (Could add a constrained for it later)
    for i in range(1, n):
        y2 = bottleneck(y2, split_size, shortcut)
        # concat_list.append(y2)
    
    # add ability to concat to receive a list, becauase the the concatenation has to be done for each bottleneck layer that was run
    var y: Symbol
    if n > 1:
        y = g.concat(split[0], split[1], y1, y2, dim=1)
    else: 
        y = g.concat(split[0], split[1], y1, dim=1)

    return Conv(g, y, weight, bias, 1, 0, 1)


fn SPPF(inout g: Graph, x: Symbol, out_channels: Int) -> Symbol:
    var conv = Conv(g, x, out_channels // 2, 1, 0, 1)

    var maxpool2d_1 = nn.MaxPool2d(g, conv, kernel_size=5, stride=StaticIntTuple[2](1), padding=2)
    var maxpool2d_2 = nn.MaxPool2d(g, maxpool2d_1, kernel_size=5, stride=StaticIntTuple[2](1), padding=2)
    var maxpool2d_3 = nn.MaxPool2d(g, maxpool2d_2, kernel_size=5, stride=StaticIntTuple[2](1), padding=2)

    var y = g.concat(conv, maxpool2d_1, maxpool2d_2, maxpool2d_3, dim=1)

    return Conv(g, y, out_channels, 1, 0, 1)


fn Detect(inout g: Graph, x: Symbol, out_channels: Int, nc: Int, detect_conv: Int) -> Symbol:
    # self.nc = nc  # number of classes
    # self.nl = len(ch)  # number of detection layers
    # self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    # self.no = nc + self.reg_max * 4  # number of outputs per anchor

    var reg_max = 16

    var c2 = max(max(16, out_channels // 4),  reg_max * 4)
    var c3 = max(0, nc)  # channels
    
    if detect_conv == 1:
        var conv1 = Conv(g, x, c2, 3, 1, 1)
        var conv1_2 = Conv(g, conv1, c2, 3, 1, 1)
        var conv1_3 = nn.Conv2d(g, conv1_2, 4 * reg_max, 1, 0, 1)

        return conv1_3
    else:
        var conv2 = Conv(g, x, c3, 3, 1, 1)
        var conv2_2 = Conv(g, conv2, c3, 3, 1, 1)
        var conv2_3 = nn.Conv2d(g, conv2_2, nc, 1, 0, 1)

        return conv2_3


fn YoloV8(batch_size: Int, yolo_model_type: StaticTuple[Float64, 3]) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 3, 640, 640))

    # Adapted from https://private-user-images.githubusercontent.com/27466624/239739723-57391d0f-1848-4388-9f30-88c2fb79233f.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTUxMTk0MDYsIm5iZiI6MTcxNTExOTEwNiwicGF0aCI6Ii8yNzQ2NjYyNC8yMzk3Mzk3MjMtNTczOTFkMGYtMTg0OC00Mzg4LTlmMzAtODhjMmZiNzkyMzNmLmpwZz9Y>LUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTA3VDIxNTgyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNlZTdkY2ZiMDA0Y2VlOGZkYjllN2FkYTQ1MTY5OWY1YzYwNjIxZDM4OTZiYWRiMGU5YWQxNzkyMTcwNGNmNTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.0ocPCiokkivvk95bQCds6Nt0EblUrHZElycV311ImF4. Some values (output_channels, stride, etc..) are different in the onnx file and the graph image.

    # Backbone
    var out_channels_1 = int(64 * yolo_model_type[1])
    var conv_1 = Conv(g, x, out_channels_1, 3, 1, 2)
    var out_channels_2 = int(128 * yolo_model_type[1])
    var conv_2 = Conv(g, conv_1, out_channels_2, 3, 1, 2)
    var C2F_n_1 = int((3 * yolo_model_type[0]) + 1) # ceil
    var C2f_1 = C2f(g, conv_2, out_channels_2, n=C2F_n_1, shortcut=True)
    var out_channels_3 = int(256 * yolo_model_type[1])
    var conv_3 = Conv(g, C2f_1, out_channels_3, 3, 1, 2)
    var C2F_n_2 = int((6 * yolo_model_type[0]) + 1) # ceil
    var C2f_2 = C2f(g, conv_3, out_channels_3, n=C2F_n_2, shortcut=True)

    var out_channels_4 = int(512 * yolo_model_type[1])
    var conv_4 = Conv(g, C2f_2, out_channels_4, 3, 1, 2)
    var C2f_3 = C2f(g, conv_4, out_channels_4, n=C2F_n_2, shortcut=True)

    var out_channels_5 = int(512 * yolo_model_type[1] * yolo_model_type[2])
    var conv_5 = Conv(g, C2f_3, out_channels_5, 3, 1, 2)
    var C2f_4 = C2f(g, conv_5, out_channels_5, n=C2F_n_1, shortcut=True)
    var SPPF_1 = SPPF(g, C2f_4, out_channels_5)

    # Head
    var upsample_1 = g.op(OP.UPSAMPLE, SPPF_1, attributes=AttributeVector(Attribute("mode", "nearest"), Attribute("scales", TensorShape(2, 2))))

    # The order of concats was wrong
    var concat_1 = g.concat(upsample_1, C2f_3, dim=1)

    var out_channels_6 = int(512 * yolo_model_type[1])
    var C2f_5 = C2f(g, concat_1, out_channels_6, n=C2F_n_1, shortcut=False)
    
    var upsample_2 = g.op(OP.UPSAMPLE, C2f_5, attributes=AttributeVector(Attribute("mode", "nearest"), Attribute("scales", TensorShape(2, 2))))

    var concat_2 = g.concat(upsample_2, C2f_2, dim=1)

    var out_channels_7 = int(256 * yolo_model_type[1])
    var C2f_6 = C2f(g, concat_2, out_channels_7, n=C2F_n_1, shortcut=False)
    
    var conv_6 = Conv(g, C2f_6, out_channels_7, 3, 1, 2)
    var concat_3 = g.concat(conv_6, C2f_5, dim=1)
    var C2f_7 = C2f(g, concat_3, out_channels_6, n=C2F_n_1, shortcut=False)

    var conv_7 = Conv(g, C2f_7, out_channels_6, 3, 1, 2)
    var concat_4 = g.concat(conv_7, SPPF_1, dim=1)
    var out_channels_8 = int(512 * yolo_model_type[1] * yolo_model_type[2])
    var C2f_8 = C2f(g, concat_4, out_channels_8, n=C2F_n_1, shortcut=False)

    # Detect
    # declare them this way because the order of initializers in the onnx file is like this
    var detect_1 = Detect(g, C2f_6, out_channels_7, 80, 1)
    var detect_2 = Detect(g, C2f_7, out_channels_6, 80, 1)
    var detect_3 = Detect(g, C2f_8, out_channels_8, 80, 1)

    var detect_1_1 = Detect(g, C2f_6, out_channels_7, 80, 2)
    var detect_2_1 = Detect(g, C2f_7, out_channels_6, 80, 2)
    var detect_3_1 = Detect(g, C2f_8, out_channels_8, 80, 2)

    var concat_detect_1 = g.concat(detect_1, detect_1_1, dim=1)
    var concat_detect_2 = g.concat(detect_2, detect_2_1, dim=1)
    var concat_detect_3 = g.concat(detect_3, detect_3_1, dim=1)

    # -------- output
    var reshape_1 = g.op(OP.RESHAPE, concat_detect_1, attributes=AttributeVector(Attribute("shape", TensorShape(1, 144, concat_detect_1.shape[2] * concat_detect_1.shape[3]))))

    var reshape_2 = g.op(OP.RESHAPE, concat_detect_2, attributes=AttributeVector(Attribute("shape", TensorShape(1, 144, concat_detect_2.shape[2] * concat_detect_2.shape[3]))))

    var reshape_3 = g.op(OP.RESHAPE, concat_detect_3, attributes=AttributeVector(Attribute("shape", TensorShape(1, 144, concat_detect_3.shape[2] * concat_detect_3.shape[3]))))

    # --

    var concat_5 = g.concat(reshape_1, reshape_2, reshape_3, dim=2)

    var split_sections = List[Int](64, 80)
    var split_1 = g.split(concat_5, split_sections, dim=1)

    var for_second_concat = g.op(OP.SIGMOID, split_1[1])

    var reshape_4 = g.op(OP.RESHAPE, split_1[0], attributes=AttributeVector(Attribute("shape", TensorShape(1, 4, 16, 8400))))

    var transpose_1 = g.op(OP.TRANSPOSE, reshape_4, attributes=AttributeVector(Attribute("axes", List[Int](0, 2, 1, 3))))

    var softmax = nn.Softmax(g, transpose_1, axis=1)

    var conv_norm_1 = nn.Conv2d(g, softmax, 1, 1, 0, 1, 1)

    var reshape_5 = g.op(OP.RESHAPE, conv_norm_1, attributes=AttributeVector(Attribute("shape", TensorShape(1, 4, 8400))))

    var slice_1 = g.op(OP.SLICE, reshape_5, attributes=AttributeVector(
        Attribute("axes", List[Int](1)), 
        Attribute("starts", List[Int](0)), 
        Attribute("ends", List[Int](2))))
    var slice_2 = g.op(OP.SLICE, reshape_5, attributes=AttributeVector(
        Attribute("axes", List[Int](1)), 
        Attribute("starts", List[Int](2)), 
        Attribute("ends", List[Int](4))))

    var sub_constant_value = g.input(TensorShape(1, 2, 8400))
    var sub_with_constant_1 = g.op(OP.SUB, sub_constant_value, slice_1)
    var add_constant_value = g.input(TensorShape(1, 2, 8400))
    var add_with_constant_2 = g.op(OP.ADD, add_constant_value, slice_2)

    var add_1 = g.op(OP.ADD, sub_with_constant_1, add_with_constant_2)
    var sub_1 = g.op(OP.SUB, add_with_constant_2, sub_with_constant_1)

    var div_1 = g.op(OP.DIV, add_1, 2)

    var concat_6 = g.concat(div_1, sub_1, dim=1)

    var mul_constant_value = g.input(TensorShape(1, 8400))
    var mul_with_constant_1 = g.op(OP.MUL, concat_6, mul_constant_value)

    var concat_7 = g.concat(mul_with_constant_1, for_second_concat, dim=1)
    
    g.out(concat_7)

    return g ^


alias yolov8_n = StaticTuple[Float64, 3](
    0.33, 0.25, 2
)  # d (depth_multiplier), w (width_multiplier), r (ratio)
# var yolov8_s
# var yolov8_m


fn get_constant_values_from_onnx_model(model_path: String) raises -> List[Tensor[dtype]]:
    var onnx = Python.import_module("onnx")

    var model = onnx.load(model_path)

    var result = List[Tensor[dtype]]()
    
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == 'value':
                    var tensor = onnx.numpy_helper.to_array(attr.t)
                    if node.name == "/model.22/Constant_9":
                        result.append(to_tensor(tensor))
                    if node.name == "/model.22/Constant_10":
                        result.append(to_tensor(tensor))
                    if node.name == "/model.22/Constant_12":
                        result.append(to_tensor(tensor))

    return result
    

fn main() raises:
    alias graph = YoloV8(1, yolov8_n)
    var model = nn.Model[graph]()

    # try: graph.render("node")
    # except: print("Could not render graph")


    model.load_model_data("./examples/data/yolov8n.onnx")

    var constant_values = get_constant_values_from_onnx_model("./examples/data/yolov8n.onnx")

    Python.add_to_path("./examples")
    var yolo_utils = Python.import_module("yolo_v8_utils")
    var image_tensor = to_tensor(yolo_utils.get_model_input('./examples/data/bus.jpg'))

    var res = model.inference(image_tensor, constant_values[0], constant_values[1], constant_values[2])

    yolo_utils.draw_bbox_from_image("./examples/data/bus.jpg", to_numpy(res[0]))