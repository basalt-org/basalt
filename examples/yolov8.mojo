from basalt.nn import Model, Conv2D, Sigmoid
from basalt.autograd import Graph

fn YoloV8(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 3, 640, 640))

    var x_conv_1 = Conv2d(g, x, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), dilation=(1, 1))
    var x_sig_1 = Sigmoid(g, x_conv_1)
    var x1 = g.op(OP.MUL, x_conv1, x_sig_1)

    var x_conv_2 = Conv2d(g, x1, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), dilation=(1, 1))
    var x_sig_2 = Sigmoid(g, x_conv_2)
    var x2 = g.op(OP.MUL, x_conv2, x_sig_2)

    var x_conv_3 = Conv2d(g, x2, out_channels=32, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), dilation=(1, 1))
    var x_sig_3 = Sigmoid(g, x_conv_3)
    var x3 = g.op(OP.MUL, x_conv3, x_sig_3)

    var xy = g.split(x3, sections=List[Int](2, 2), dim=1)
    var x_x = xy[0]
    var x_y = xy[1]

    var x_y_conv1 = Conv2d(g, x_y, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1))
    var x_y_sig1 = Sigmoid(g, x_y_conv1)
    var x_y1 = g.op(OP.MUL, x_y_conv1, x_y_sig1)

    var x_y_conv2 = Conv2d(g, x_y1, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1))
    var x_y_sig2 = Sigmoid(g, x_y_conv2)
    var x_y2 = g.op(OP.MUL, x_y_conv2, x_y_sig2)

    var x_y3 = g.op(OP.ADD, x_y, x_y2)

    var z = g.concat(x_x, x_y3, dim=1)

    var z_conv_1 = Conv2d(g, z, out_channels=32, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), dilation=(1, 1))
    var z_sig_1 = Sigmoid(g, z_conv_1)
    var z1 = g.op(OP.MUL, z_conv1, z_sig_1)

    var z_conv_2 = Conv2d(g, z1, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), dilation=(1, 1))
    var z_sig_2 = Sigmoid(g, z_conv_2)
    var z2 = g.op(OP.MUL, z_conv2, z_sig_2)

    var z_conv_3 = Conv2d(g, z2, out_channels=64, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), dilation=(1, 1))
    var z_sig_3 = Sigmoid(g, z_conv_3)
    var z3 = g.op(OP.MUL, z_conv3, z_sig_3)

    var zy = g.split(z3, sections=List[Int](2, 2), dim=1)
    var z_x = zy[0]
    var z_y = zy[1]

    var z_y_conv1 = Conv2d(g, z_y, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1))
    var z_y_sig1 = Sigmoid(g, z_y_conv1)
    var z_y1 = g.op(OP.MUL, z_y_conv1, z_y_sig1)

    var z_y_conv2 = Conv2d(g, z_y1, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1))
    var z_y_sig2 = Sigmoid(g, z_y_conv2)
    var z_y2 = g.op(OP.MUL, z_y_conv2, z_y_sig2)

    var z_y3 = g.op(OP.ADD, z_y, z_y2)

    var z_y_conv_3 = Conv2d(g, z_y3, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1))
    var z_y_sig_3 = Sigmoid(g, z_y_conv_3)
    var z_y4 = g.op(OP.MUL, z_y_conv_3, z_y_sig_3)

    var z_y_conv_4 = Conv2d(g, z_y4, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1))
    var z_y_sig_4 = Sigmoid(g, z_y_conv_4)
    var z_y5 = g.op(OP.MUL, z_y_conv_4, z_y_sig_4)

    var z_y6 = g.op(OP.ADD, z_y3, z_y5)

    var z_y7 = g.concat(z_x, z_y3, z_y6, dim=1)

    return g ^


fn main():
    alias graph = YoloV8()
    var model = nn.Model[graph]()
    ...