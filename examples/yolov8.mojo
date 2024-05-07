from basalt.nn import Model, Conv2D, Sigmoid
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

    return g ^


fn main():
    alias graph = YoloV8()
    var model = nn.Model[graph]()
    ...