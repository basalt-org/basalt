from python.python import Python
from tensor import Tensor, TensorShape
from testing import assert_equal
from random import rand

from dainemo import GRAPH
from dainemo.autograd.ops.pool import MAXPOOL2D
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


fn test_forward_1() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 1, 28, 28)  kernel size: (5, 5)
    alias kernel_size = 5
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    var inputs = rand[dtype](4, 1, 28, 28)

    var res = MAXPOOL2D.forward[kernel_size, stride, padding, dilation](inputs)
    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=res.grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()


fn test_forward_2() raises:
    # padding=0, stride=1, dilation=1
    # input shape: (4, 1, 32, 17)  kernel size: (2, 2)
    alias kernel_size = StaticIntTuple[2](2, 2)
    alias padding = 0
    alias stride = 1
    alias dilation = 1
    var inputs = rand[dtype](4, 1, 32, 17)
    
    var res = MAXPOOL2D.forward[kernel_size, stride, padding, dilation](inputs)
    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=res.grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()



fn test_forward_3() raises:
    # padding=(3, 1), stride=(2, 3), dilation=(2, 3)
    # input shape: (4, 3, 32, 17)  kernel size: (6, 6)
    alias kernel_size = StaticIntTuple[2](6, 6)
    alias padding = StaticIntTuple[2](3, 1)
    alias stride = StaticIntTuple[2](2, 3)
    alias dilation = StaticIntTuple[2](2, 3)
    var inputs = Tensor[dtype](4, 3, 32, 17)
    fill[dtype, nelts](inputs, 1.0)

    var res = MAXPOOL2D.forward[kernel_size, stride, padding, dilation](inputs)
    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=res.grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()


fn test_backward_1() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 1, 28, 28)  kernel size: (5, 5)
    alias kernel_size = 5
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    var inputs = rand[dtype](4, 1, 28, 28)
    
    var res = MAXPOOL2D.forward[kernel_size, stride, padding, dilation](inputs)
    
    var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)
    var upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())
    
    var ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=upper_grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    assert_tensors_equal(ug1, torch_out.expected_grad, "almost")
    GRAPH.reset_all()


fn test_backward_2() raises:
    # padding=0, stride=1, dilation=1
    # input shape: (4, 1, 32, 17)  kernel size: (2, 2)
    alias kernel_size = 2
    alias padding = 0
    alias stride = 1
    alias dilation = 1
    var inputs = rand[dtype](4, 1, 32, 17)

    var res = MAXPOOL2D.forward[kernel_size, stride, padding, dilation](inputs)
    
    var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)
    var upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

    var ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=upper_grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    assert_tensors_equal(ug1, torch_out.expected_grad, "almost")
    GRAPH.reset_all()


fn test_backward_3() raises:
    # padding=(3, 1), stride=(2, 3), dilation=(2, 3)
    # input shape: (4, 3, 32, 17)  kernel size: (6, 6)
    alias kernel_size = StaticIntTuple[2](6, 6)
    alias padding = StaticIntTuple[2](3, 1)
    alias stride = StaticIntTuple[2](2, 3)
    alias dilation = StaticIntTuple[2](2, 3)
    var inputs = Tensor[dtype](4, 3, 32, 17)
    fill[dtype, nelts](inputs, 1.0)

    var res = MAXPOOL2D.forward[kernel_size, stride, padding, dilation](inputs)
    
    var gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)
    var upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

    var ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    var torch_out = torch_maxpool2d(
        inputs,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=upper_grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    assert_tensors_equal(ug1, torch_out.expected_grad, "almost")
    GRAPH.reset_all()


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