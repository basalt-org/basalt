from python.python import Python
from tensor import Tensor, TensorShape
from testing import assert_equal
from random import rand

from dainemo import GRAPH
from dainemo.autograd.ops.pool import MAXPOOL2D
from test_conv import to_numpy, to_tensor
from test_tensorutils import assert_tensors_equal


alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


@value
struct torch_maxpool2d_output:
    var expected: Tensor[dtype]
    var expected_grad: Tensor[dtype]
    
fn torch_maxpool2d(inputs: Tensor[dtype], kernel_shape: TensorShape, padding: Int, stride: Int, upper_grad: Tensor) -> torch_maxpool2d_output:
    let out: torch_maxpool2d_output
    
    try:
        let torch = Python.import_module("torch")
        let F = Python.import_module("torch.nn.functional")
        let np = Python.import_module("numpy")

        let inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)

        let expected = F.max_pool2d(
            inputs,
            (kernel_shape[-2], kernel_shape[-1]),
            stride,
            padding
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


fn test_forward() raises:
    alias padding = 2
    alias stride = 1
    alias batch = 4
    alias in_channels = 3
    alias kernel_shape = TensorShape(in_channels, in_channels, 5, 5)
    let inputs = rand[dtype](batch, in_channels, 28, 28)

    let res = MAXPOOL2D.forward[kernel_shape, padding, stride](inputs)
    let torch_out = torch_maxpool2d(inputs, kernel_shape, padding=padding, stride=stride, upper_grad=res.grad)
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()

fn test_backward() raises:
    alias padding = 2
    alias stride = 1
    alias batch = 4
    alias in_channels = 3
    alias kernel_shape = TensorShape(in_channels, in_channels, 10, 10)
    let inputs = rand[dtype](batch, in_channels, 28, 28)
    

    let res = MAXPOOL2D.forward[kernel_shape, padding, stride](inputs)
    
    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 1)
    let upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())
    
    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    let torch_out = torch_maxpool2d(inputs, kernel_shape, padding=padding, stride=stride, upper_grad=upper_grad)
    assert_tensors_equal(res.tensor, torch_out.expected)
    assert_tensors_equal(ug1, torch_out.expected_grad, "almost")

    GRAPH.reset_all()

fn main():

    try:
        test_forward()
        test_backward()
    except:
        print("Error")