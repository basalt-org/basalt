from python.python import Python
from tensor import TensorShape
from testing import assert_equal
from random import rand

import dainemo.nn as nn
from dainemo import Graph, Symbol, OP
from dainemo.autograd.ops.conv import get_result_shape
from dainemo.autograd.ops.pool import MAXPOOL2D
from dainemo.autograd.attributes import Attribute, AttributeVector
from test_conv import to_numpy, to_tensor
from test_tensorutils import assert_tensors_equal, fill


alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


@value
struct torch_maxpool2d_output:
    var expected: Tensor[dtype]
    var expected_grad: Tensor[dtype]
    
fn torch_maxpool2d(
    inputs: Tensor,
    kernel_size: StaticIntTuple[2],
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
    upper_grad: Tensor
) -> torch_maxpool2d_output:
    var out: torch_maxpool2d_output
    
    try:
        var torch = Python.import_module("torch")
        var F = Python.import_module("torch.nn.functional")
        var np = Python.import_module("numpy")

        var inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)

        var expected = F.max_pool2d(
            inputs,
            (kernel_size[0], kernel_size[1]),
            (stride[0], stride[1]),
            (padding[0], padding[1]),
            (dilation[0], dilation[1]),
        )

        # uppergrad & backwards
        var upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        # expected
        out = torch_maxpool2d_output(
            to_tensor(expected.detach().numpy()),
            to_tensor(inputs.grad.numpy())
        )
        return out

    except:
        print("Error in torch_maxpool2d")
        var d = Tensor[dtype](1)
        var out = torch_maxpool2d_output(d, d)
        return out


fn test_pool_forward[
    input_shape: TensorShape,
    kernel_size: StaticIntTuple[2],
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2]
](
    inputs: Tensor[dtype]
) raises:

    fn create_graph() -> Graph:
        var g = Graph()
        var inp = g.input(input_shape)

        var res = g.op(OP.MAXPOOL2D, inp, attributes=AttributeVector(
            Attribute("kernel_size", kernel_size),
            Attribute("padding", padding),
            Attribute("stride", stride),
            Attribute("dilation", dilation)
        ))
        _ = g.out(res)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph]()
    var res = model.forward(inputs)

    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=Tensor[dtype](res.shape())
    )

    assert_tensors_equal(res, torch_out.expected)


fn test_forward_1() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 1, 28, 28)  kernel size: (5, 5)
    alias kernel_size = 5
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    alias input_shape = TensorShape(4, 1, 28, 28)
    var inputs = rand[dtype](input_shape)

    test_pool_forward[input_shape, kernel_size, padding, stride, dilation](inputs)


fn test_forward_2() raises:
    # padding=0, stride=1, dilation=1
    # input shape: (4, 1, 32, 17)  kernel size: (2, 2)
    alias kernel_size = StaticIntTuple[2](2, 2)
    alias padding = 0
    alias stride = 1
    alias dilation = 1
    alias input_shape = TensorShape(4, 1, 32, 17)
    var inputs = rand[dtype](input_shape)
    
    test_pool_forward[input_shape, kernel_size, padding, stride, dilation](inputs)


fn test_forward_3() raises:
    # padding=(3, 1), stride=(2, 3), dilation=(2, 3)
    # input shape: (4, 3, 32, 17)  kernel size: (6, 6)
    alias kernel_size = StaticIntTuple[2](6, 6)
    alias padding = StaticIntTuple[2](3, 1)
    alias stride = StaticIntTuple[2](2, 3)
    alias dilation = StaticIntTuple[2](2, 3)
    alias input_shape = TensorShape(4, 3, 32, 17)
    var inputs = rand[dtype](input_shape)

    test_pool_forward[input_shape, kernel_size, padding, stride, dilation](inputs)



fn test_pool_backward[
    ug_shape: TensorShape,
    input_shape: TensorShape,
    kernel_size: StaticIntTuple[2],
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2]
](
    ug: Tensor[dtype], inputs: Tensor[dtype]
) raises:

    alias attributes = AttributeVector(
        Attribute("kernel_size", kernel_size),
        Attribute("padding", padding),
        Attribute("stride", stride),
        Attribute("dilation", dilation)
    )
    
    var grad = MAXPOOL2D.backward[ug_shape, input_shape, attributes](ug, inputs)

    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=ug
    )

    assert_tensors_equal(grad, torch_out.expected_grad, "almost")


fn test_backward_1() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 1, 28, 28)  kernel size: (5, 5)
    alias kernel_size = 5
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    alias input_shape = TensorShape(4, 1, 28, 28)
    var inputs = rand[dtype](input_shape)

    # uppergrad
    alias kernel_size_static: StaticIntTuple[2] = kernel_size
    alias res = get_result_shape(input_shape, TensorShape(kernel_size_static), padding, stride, dilation)
    alias ug_shape = TensorShape(input_shape[0], input_shape[1], res[0], res[1])
    var ug = rand[dtype](ug_shape)

    test_pool_backward[ug_shape, input_shape, kernel_size, padding, stride, dilation](ug, inputs)


fn test_backward_2() raises:
    # padding=0, stride=1, dilation=1
    # input shape: (4, 1, 32, 17)  kernel size: (2, 2)
    alias kernel_size = 2
    alias padding = 0
    alias stride = 1
    alias dilation = 1
    alias input_shape = TensorShape(4, 1, 32, 17)
    var inputs = rand[dtype](input_shape)

    # uppergrad
    alias kernel_size_static: StaticIntTuple[2] = kernel_size
    alias res = get_result_shape(input_shape, TensorShape(kernel_size_static), padding, stride, dilation)
    alias ug_shape = TensorShape(input_shape[0], input_shape[1], res[0], res[1])
    var ug = rand[dtype](ug_shape)

    test_pool_backward[ug_shape, input_shape, kernel_size, padding, stride, dilation](ug, inputs)


fn test_backward_3() raises:
    # padding=(3, 1), stride=(2, 3), dilation=(2, 3)
    # input shape: (4, 3, 32, 17)  kernel size: (6, 6)
    alias kernel_size = StaticIntTuple[2](6, 6)
    alias padding = StaticIntTuple[2](3, 1)
    alias stride = StaticIntTuple[2](2, 3)
    alias dilation = StaticIntTuple[2](2, 3)
    alias input_shape = TensorShape(4, 3, 32, 17)
    var inputs = rand[dtype](input_shape)

    # uppergrad
    alias kernel_size_static: StaticIntTuple[2] = kernel_size
    alias res = get_result_shape(input_shape, TensorShape(kernel_size_static), padding, stride, dilation)
    alias ug_shape = TensorShape(input_shape[0], input_shape[1], res[0], res[1])
    var ug = rand[dtype](ug_shape)

    test_pool_backward[ug_shape, input_shape, kernel_size, padding, stride, dilation](ug, inputs)


fn main():
    try:
        test_forward_1()
        test_forward_2()
        test_forward_3()
        test_backward_1()
        test_backward_2()
        test_backward_3()
    except e:
        print("[Error] Error in MaxPool2D")
        print(e)