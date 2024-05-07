from basalt.nn import Model, Conv2D, Sigmoid, MaxPool2d
from basalt.autograd import Graph, Symbol

fn YoloV8(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 3, 640, 640))

    @parameter
    fn CSM(x: Symbol, out_channels: Int, kernel_size: Int, padding: Int, stride: Int, dilation: Int) -> Symbol:
        return g.op(
            OP.MUL,
            Conv2D(g, x, out_channels, kernel_size, padding, stride, dilation),
            Sigmoid(g, conv),
        )

    var x1 = CSM(x, 16, 3, 1, 2, 1)
    var x2 = CSM(x1, 32, 3, 1, 2, 1)
    var x3 = CSM(x2, 32, 1, 0, 1, 1)

    var x_split_1 = g.split(x3, 2, 1)
    var x1_x = x_split_1[0]
    var x1_y = x_split_1[1]
    
    var x1_y1 = CSM(x1_y, 16, 3, 1, 1, 1)
    var x1_y2 = CSM(x1_y1, 16, 3, 1, 1, 1)
    var x1_y3 = g.op(OP.ADD, x1_y, x1_y2)

    var x4 = g.concat(x1_x, x1_y, x1_y3, 1)
    var x5 = CSM(x4, 32, 1, 0, 1, 1)
    var x6 = CSM(x5, 64, 3, 1, 2, 1)
    var x7 = CSM(x6, 64, 1, 0, 1, 1)

    var x_split_2 = g.split(x7, 2, 1)
    var x2_x = x_split_2[0]
    var x2_y = x_split_2[1]

    var x2_y1 = CSM(x2_y, 32, 3, 1, 1, 1)
    var x2_y2 = CSM(x2_y1, 32, 3, 1, 1, 1)
    var x2_y3 = g.op(OP.ADD, x2_y, x2_y2)
    var x2_y4 = CSM(x2_y3, 32, 3, 1, 1, 1)
    var x2_y5 = CSM(x2_y4, 32, 3, 1, 1, 1)
    var x2_y6 = g.op(OP.ADD, x2_y3, x2_y5)

    var x8 = g.concat(x2_x, x2_y, x2_y3, x2_y6, 1)
    var x9 = CSM(x8, 64, 1, 0, 1, 1)
    var x10 = CSM(x9, 128, 3, 1, 2, 1)
    var x11 = CSM(x10, 128, 1, 0, 1, 1)

    var x_split_3 = g.split(x11, 2, 1)
    var x3_x = x_split_3[0]
    var x3_y = x_split_3[1]

    var x3_y1 = CSM(x3_y, 64, 3, 1, 1, 1)
    var x3_y2 = CSM(x3_y1, 64, 3, 1, 1, 1)
    var x3_y3 = g.op(OP.ADD, x3_y, x3_y2)
    var x3_y4 = CSM(x3_y3, 64, 3, 1, 1, 1)
    var x3_y5 = CSM(x3_y4, 64, 3, 1, 1, 1)
    var x3_y6 = g.op(OP.ADD, x3_y3, x3_y5)

    var x12 = g.concat(x3_x, x3_y, x3_y3, x3_y6, 1)
    var x13 = CSM(x12, 128, 1, 0, 1, 1)
    var x14 = CSM(x13, 256, 3, 1, 2, 1)
    var x15 = CSM(x14, 256, 1, 0, 1, 1)
    
    var x_split_4 = g.split(x15, 2, 1)
    var x4_x = x_split_4[0]
    var x4_y = x_split_4[1]

    var x4_y1 = CSM(x4_y, 128, 3, 1, 1, 1)
    var x4_y2 = CSM(x4_y1, 128, 3, 1, 1, 1)
    var x4_y3 = g.op(OP.ADD, x4_y, x4_y2)

    var x16 = g.concat(x4_x, x4_y, x4_y3, 1)
    var x17 = CSM(x16, 256, 1, 0, 1, 1)
    var x18 = CSM(x17, 128, 1, 0, 1, 1)

    var x19 = MaxPool2d(g, x18, 5, 1, 2, 1)
    var x20 = MaxPool2d(g, x19, 5, 1, 2, 1)
    var x21 = MaxPool2d(g, x20, 5, 1, 2, 1)
    var x22 = g.concat(x18, x19, x20, x21, 1)
    var x23 = CSM(x22, 256, 1, 0, 1, 1)

    # NOTE: This is where the "Resize: 4" OP is
 
    return g ^


fn main():
    alias graph = YoloV8()
    var model = nn.Model[graph]()
    ...