from basalt.nn import Model, Conv2D, Sigmoid
from basalt.autograd import Graph

fn YoloV8(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 3, 640, 640))

    var x_conv_1 = Conv2d(g, x, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), dilation=(1, 1))
    var x_sig_1 = Sigmoid(g, x_conv_1)
    g.op(OP.MUL, x_conv1, x_sig_1)

    var x_conv_2 = Conv2d(g, x, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), dilation=(1, 1))
    var x_sig_2 = Sigmoid(g, x_conv_2)
    g.op(OP.MUL, x_conv2, x_sig_2)

    var x_conv_3 = Conv2d(g, x, out_channels=32, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), dilation=(1, 1))
    var x_sig_3 = Sigmoid(g, x_conv_3)
    g.op(OP.MUL, x_conv3, x_sig_3)

    ...

    return g ^


fn main():
    alias graph = YoloV8()
    var model = nn.Model[graph]()
    ...