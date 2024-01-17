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
    kernel_shape: TensorShape,
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
    upper_grad: Tensor
) -> torch_maxpool2d_output:
    let out: torch_maxpool2d_output
    
    try:
        let torch = Python.import_module("torch")
        let F = Python.import_module("torch.nn.functional")
        let np = Python.import_module("numpy")

        let inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)

        let expected = F.max_pool2d(
            inputs,
            (kernel_shape[-2], kernel_shape[-1]),
            (stride[0], stride[1]),
            (padding[0], padding[1]),
            (dilation[0], dilation[1]),
        )

        # uppergrad & backwards
        let upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        # expected
        out = torch_maxpool2d_output(
            to_tensor(expected.detach().numpy()),
            to_tensor(inputs.grad.numpy())
        )
        return out

    except:
        print("Error in torch_maxpool2d")
        let d = Tensor[dtype](1)
        let out = torch_maxpool2d_output(d, d)
        return out


fn test_forward_1() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 1, 28, 28)  kernel shape: (1, 1, 5, 5)
    alias kernel_shape = TensorShape(1, 1, 5, 5)
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    let inputs = rand[dtype](4, 1, 28, 28)

    let res = MAXPOOL2D.forward[kernel_shape, padding, stride, dilation](inputs)
    let torch_out = torch_maxpool2d(
        inputs,
        kernel_shape,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=res.grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()


fn test_forward_2() raises:
    # padding=0, stride=1, dilation=1
    # input shape: (4, 1, 32, 17)  kernel shape: (1, 1, 2, 2)
    alias kernel_shape = TensorShape(1, 1, 2, 2)
    alias padding = 0
    alias stride = 1
    alias dilation = 1
    let inputs = rand[dtype](4, 1, 32, 17)
    
    let res = MAXPOOL2D.forward[kernel_shape, padding, stride, dilation](inputs)
    let torch_out = torch_maxpool2d(
        inputs,
        kernel_shape,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=res.grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()



fn test_forward_3() raises:
    # padding=(3, 1), stride=(2, 3), dilation=(2, 3)
    # input shape: (4, 3, 32, 17)  kernel shape: (2, 3, 2, 2)
    alias kernel_shape = TensorShape(3, 3, 6, 6)
    alias padding = StaticIntTuple[2](3, 1)
    alias stride = StaticIntTuple[2](2, 3)
    alias dilation = StaticIntTuple[2](2, 3)
    var inputs = Tensor[dtype](4, 3, 32, 17)
    fill[dtype, nelts](inputs, 1.0)

    let res = MAXPOOL2D.forward[kernel_shape, padding, stride, dilation](inputs)
    let torch_out = torch_maxpool2d(
        inputs,
        kernel_shape,
        padding=padding,
        stride=stride,
        dilation=dilation,
        upper_grad=res.grad
    )
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()


fn test_backward_1() raises:
    # padding=2, stride=1, dilation=1
    # input shape: (4, 1, 28, 28)  kernel shape: (1, 1, 5, 5)
    alias kernel_shape = TensorShape(1, 1, 5, 5)
    alias padding = 2
    alias stride = 1
    alias dilation = 1
    let inputs = rand[dtype](4, 1, 28, 28)
    
    let res = MAXPOOL2D.forward[kernel_shape, padding, stride, dilation](inputs)
    
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)
    let upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())
    
    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    let torch_out = torch_maxpool2d(
        inputs,
        kernel_shape,
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
    # input shape: (4, 1, 32, 17)  kernel shape: (1, 1, 2, 2)
    alias kernel_shape = TensorShape(1, 1, 2, 2)
    alias padding = 0
    alias stride = 1
    alias dilation = 1
    let inputs = rand[dtype](4, 1, 32, 17)

    let res = MAXPOOL2D.forward[kernel_shape, padding, stride, dilation](inputs)
    
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)
    let upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    let torch_out = torch_maxpool2d(
        inputs,
        kernel_shape,
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
    # input shape: (4, 3, 32, 17)  kernel shape: (2, 3, 2, 2)
    alias kernel_shape = TensorShape(3, 3, 6, 6)
    alias padding = StaticIntTuple[2](3, 1)
    alias stride = StaticIntTuple[2](2, 3)
    alias dilation = StaticIntTuple[2](2, 3)
    var inputs = Tensor[dtype](4, 3, 32, 17)
    fill[dtype, nelts](inputs, 1.0)

    let res = MAXPOOL2D.forward[kernel_shape, padding, stride, dilation](inputs)
    
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)
    let upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    let torch_out = torch_maxpool2d(
        inputs,
        kernel_shape,
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