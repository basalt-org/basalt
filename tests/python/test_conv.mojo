from random import rand
from python.python import Python
from testing import assert_equal

from basalt import dtype, nelts
from basalt.autograd import Graph, Symbol
from basalt.autograd.attributes import Attribute, AttributeVector
from basalt.autograd.ops import OP
from basalt.autograd.ops.conv import get_result_shape, CONV2D
from basalt.nn import Tensor, TensorShape, Model
from basalt.utils.tensorutils import fill

from tests import assert_tensors_equal, to_numpy, to_tensor


fn test_get_result_shape() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 28, 28)  kernel shape: (1, 16)
    # result:  (32, 17)
    var inputs = Tensor[dtype](4, 28, 28)
    var kernel = Tensor[dtype](1, 16)

    var res = get_result_shape(inputs.shape(), kernel.shape(), 2, 1, 1)
    assert_equal(res[0], 32)
    assert_equal(res[1], 17)

    # padding=0, stride=1, dilation=1
    # input shape: (4, 32, 17)  kernel shape: (2, 2)
    # result:  (31, 16)
    inputs = Tensor[dtype](4, 32, 17)
    kernel = Tensor[dtype](2, 2)

    res = get_result_shape(inputs.shape(), kernel.shape(), 0, 1, 1)
    assert_equal(res[0], 31)
    assert_equal(res[1], 16)

    # padding=(3, 1), stride=1, dilation=2
    # input shape: (4, 32, 17)  kernel shape: (2, 2)
    # result:  (36, 17)
    inputs = Tensor[dtype](4, 32, 17)
    kernel = Tensor[dtype](2, 2)

    res = get_result_shape(
        inputs.shape(), kernel.shape(), StaticIntTuple[2](3, 1), 1, 2
    )
    assert_equal(res[0], 36)
    assert_equal(res[1], 17)

    # padding=(3, 2), stride=(2, 1), dilation=(2, 3)
    # input shape: (4, 32, 17)  kernel shape: (2, 2)
    # result:  (18, 18)
    inputs = Tensor[dtype](4, 32, 17)
    kernel = Tensor[dtype](3, 2)

    res = get_result_shape(
        inputs.shape(),
        kernel.shape(),
        StaticIntTuple[2](3, 2),
        StaticIntTuple[2](2, 1),
        StaticIntTuple[2](2, 3),
    )
    assert_equal(res[0], 17)
    assert_equal(res[1], 18)


@value
struct torch_conv2d_output:
    var expected: Tensor[dtype]
    var expected_inputs_grad: Tensor[dtype]
    var expected_kernel_grad: Tensor[dtype]
    var expected_bias_grad: Tensor[dtype]


fn torch_conv2d(
    inputs: Tensor,
    kernel: Tensor,
    bias: Tensor,
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
    upper_grad: Tensor,
) -> torch_conv2d_output:
    var out: torch_conv2d_output

    try:
        var torch = Python.import_module("torch")
        var F = Python.import_module("torch.nn.functional")
        var np = Python.import_module("numpy")

        var inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)
        var weights = torch.from_numpy(to_numpy(kernel)).requires_grad_(True)
        var bias = torch.from_numpy(to_numpy(bias)).requires_grad_(True)

        var expected = F.conv2d(
            inputs,
            weights,
            bias,
            (stride[0], stride[1]),
            (padding[0], padding[1]),
            (dilation[0], dilation[1]),
        )

        # uppergrad & backwards
        var upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        # expected output
        out = torch_conv2d_output(
            to_tensor(expected.detach().numpy()),
            to_tensor(inputs.grad.numpy()),
            to_tensor(weights.grad.numpy()),
            to_tensor(bias.grad.numpy()),
        )
        return out

    except:
        print("Error importing torch")
        var d = Tensor[dtype](1)
        var out: torch_conv2d_output = torch_conv2d_output(d, d, d, d)
        return out


fn test_conv_forward[
    input_shape: TensorShape,
    kernel_shape: TensorShape,
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
](inputs: Tensor[dtype], kernel: Tensor[dtype], bias: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var inp = g.input(input_shape)

        var weights = g.input(kernel_shape)  # as input
        var bias = g.input(kernel_shape[0])  # as input

        var res = g.op(
            OP.CONV2D,
            inp,
            weights,
            bias,
            attributes=AttributeVector(
                Attribute("padding", padding),
                Attribute("stride", stride),
                Attribute("dilation", dilation),
            ),
        )
        g.out(res)

        return g^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = Model[graph](inference_only=True)
    var res = model.inference(inputs, kernel, bias)[0]

    var torch_out = torch_conv2d(
        inputs,
        kernel,
        bias=bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=Tensor[dtype](res.shape()),
    )

    assert_tensors_equal(res, torch_out.expected)


fn test_forward_1() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 1, 28, 28)  kernel shape: (1, 1, 1, 16)
    # result_shape:  (4, 1, 32, 17)
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    alias input_shape = TensorShape(4, 1, 28, 28)
    alias kernel_shape = TensorShape(1, 1, 1, 16)

    var inputs = Tensor[dtype](input_shape)
    var kernel = Tensor[dtype](kernel_shape)
    var bias = Tensor[dtype](kernel_shape[0])
    fill(inputs, 1.0)
    fill(kernel, 1.0)

    test_conv_forward[input_shape, kernel_shape, padding, stride, dilation](
        inputs, kernel, bias
    )


fn test_forward_2() raises:
    # padding=0, stride=1, dilation=1
    # input shape: (4, 1, 32, 17)  kernel shape: (1, 1, 2, 2)
    # result_shape:  (4, 1, 31, 16)
    alias padding = 0
    alias stride = 1
    alias dilation = 1
    alias input_shape = TensorShape(4, 1, 32, 17)
    alias kernel_shape = TensorShape(1, 1, 2, 2)

    var inputs = Tensor[dtype](input_shape)
    var kernel = Tensor[dtype](kernel_shape)
    fill(inputs, 1.0)
    fill(kernel, 1.0)
    var bias = Tensor[dtype](kernel_shape[0])
    fill(bias, 66.99)

    test_conv_forward[input_shape, kernel_shape, padding, stride, dilation](
        inputs, kernel, bias
    )


fn test_forward_3() raises:
    # padding=(3, 1), stride=(2, 3), dilation=(2, 3)
    # input shape: (4, 3, 32, 17)  kernel shape: (2, 3, 2, 2)
    # result_shape:  (4, 2, 18, 6)
    alias padding = StaticIntTuple[2](3, 1)
    alias stride = StaticIntTuple[2](2, 3)
    alias dilation = StaticIntTuple[2](2, 3)
    alias input_shape = TensorShape(4, 3, 32, 17)
    alias kernel_shape = TensorShape(2, 3, 2, 2)

    var inputs = Tensor[dtype](input_shape)
    var kernel = Tensor[dtype](kernel_shape)
    fill(inputs, 3.0)
    fill(kernel, 2.0)
    var bias = Tensor[dtype](kernel_shape[0])
    fill(bias, 3)

    test_conv_forward[input_shape, kernel_shape, padding, stride, dilation](
        inputs, kernel, bias
    )


fn test_conv_backward[
    ug_shape: TensorShape,
    input_shape: TensorShape,
    kernel_shape: TensorShape,
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
](
    ug: Tensor[dtype],
    inputs: Tensor[dtype],
    kernel: Tensor[dtype],
    bias: Tensor[dtype],
) raises:
    alias bias_shape = TensorShape(kernel_shape[0])
    alias attributes = AttributeVector(
        Attribute("padding", padding),
        Attribute("stride", stride),
        Attribute("dilation", dilation),
    )

    var grad1 = CONV2D.backward[
        0, ug_shape, input_shape, kernel_shape, bias_shape, attributes
    ](ug, inputs, kernel, bias)
    var grad2 = CONV2D.backward[
        1, ug_shape, input_shape, kernel_shape, bias_shape, attributes
    ](ug, inputs, kernel, bias)
    var grad3 = CONV2D.backward[
        2, ug_shape, input_shape, kernel_shape, bias_shape, attributes
    ](ug, inputs, kernel, bias)

    var torch_out = torch_conv2d(
        inputs,
        kernel,
        bias=bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=ug,
    )

    assert_tensors_equal["almost"](grad1, torch_out.expected_inputs_grad)
    assert_tensors_equal["almost"](grad2, torch_out.expected_kernel_grad)
    assert_tensors_equal["almost"](grad3, torch_out.expected_bias_grad)


fn test_backward_1() raises:
    # padding=2, stride=1, dilation=1
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    alias input_shape = TensorShape(4, 2, 28, 28)
    alias kernel_shape = TensorShape(3, 2, 1, 16)

    var inputs = Tensor[dtype](input_shape)
    var kernel = Tensor[dtype](kernel_shape)
    fill(inputs, 1.0)
    fill(kernel, 1.0)
    var bias = Tensor[dtype](kernel_shape[0])
    rand[dtype](bias.data(), bias.num_elements())

    # uppergrad
    alias res = get_result_shape(
        input_shape, kernel_shape, padding, stride, dilation
    )
    alias ug_shape = TensorShape(
        input_shape[0], kernel_shape[0], res[0], res[1]
    )
    var ug = Tensor[dtype](ug_shape)

    test_conv_backward[
        ug_shape, input_shape, kernel_shape, padding, stride, dilation
    ](ug, inputs, kernel, bias)


fn test_backward_2() raises:
    # padding=(2, 4), stride=(3, 1), dilation=2
    alias padding = StaticIntTuple[2](2, 4)
    alias stride = StaticIntTuple[2](3, 1)
    alias dilation = 2
    alias input_shape = TensorShape(4, 2, 28, 28)
    alias kernel_shape = TensorShape(3, 2, 4, 8)

    var inputs = Tensor[dtype](input_shape)
    var kernel = Tensor[dtype](kernel_shape)
    fill(inputs, 3.0)
    fill(kernel, 1.0)
    var bias = Tensor[dtype](kernel_shape[0])
    rand[dtype](bias.data(), bias.num_elements())

    # uppergrad
    alias res = get_result_shape(
        input_shape, kernel_shape, padding, stride, dilation
    )
    alias ug_shape = TensorShape(
        input_shape[0], kernel_shape[0], res[0], res[1]
    )
    var ug = Tensor[dtype](ug_shape)
    rand[dtype](ug.data(), ug.num_elements())

    test_conv_backward[
        ug_shape, input_shape, kernel_shape, padding, stride, dilation
    ](ug, inputs, kernel, bias)


fn test_backward_3() raises:
    # padding=(2, 4), stride=2, dilation=(3, 2)
    alias padding = StaticIntTuple[2](3, 2)
    alias stride = 2
    alias dilation = StaticIntTuple[2](3, 2)
    alias input_shape = TensorShape(4, 2, 28, 28)
    alias kernel_shape = TensorShape(3, 2, 5, 6)

    var inputs = Tensor[dtype](input_shape)
    var kernel = Tensor[dtype](kernel_shape)
    fill(inputs, 3.0)
    fill(kernel, 4.0)
    var bias = Tensor[dtype](kernel_shape[0])
    rand[dtype](bias.data(), bias.num_elements())

    # uppergrad
    alias res = get_result_shape(
        input_shape, kernel_shape, padding, stride, dilation
    )
    alias ug_shape = TensorShape(
        input_shape[0], kernel_shape[0], res[0], res[1]
    )
    var ug = Tensor[dtype](ug_shape)
    rand[dtype](ug.data(), ug.num_elements())

    test_conv_backward[
        ug_shape, input_shape, kernel_shape, padding, stride, dilation
    ](ug, inputs, kernel, bias)


fn main():
    try:
        test_get_result_shape()
        test_forward_1()
        test_forward_2()
        test_forward_3()
        test_backward_1()
        test_backward_2()
        test_backward_3()
    except e:
        print("[Error] Error in Conv2D")
        print(e)
