from python.python import Python
from tensor import Tensor, TensorShape
from random import rand

from dainemo import GRAPH
from dainemo.autograd.ops.max_pool import MAXPOOL2D
from test_conv import to_numpy, to_tensor
from test_tensorutils import assert_tensors_equal


alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


@value
struct torch_maxpool2d_output:
    var expected: Tensor[dtype]
    
fn torch_maxpool2d(inputs: Tensor[dtype], kernel_size: TensorShape, padding: Int, stride: Int) -> torch_maxpool2d_output:
    let out: torch_maxpool2d_output
    
    try:
        let torch = Python.import_module("torch")
        let F = Python.import_module("torch.nn.functional")
        let np = Python.import_module("numpy")

        let inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)

        let expected = F.max_pool2d(
            inputs,
            (kernel_size[-2], kernel_size[-1]),
            stride,
            padding
        )

        # expected
        out = torch_maxpool2d_output(
            to_tensor(expected)
        )
        return out

    except:
        print("Error in torch_maxpool2d")
        let d = Tensor[dtype](1)
        let out = torch_maxpool2d_output(d)
        return out


fn test_forward() raises:
    alias padding = 2
    alias stride = 1
    alias batch = 4
    alias in_channels = 3
    let inputs = rand[dtype](batch, in_channels, 28, 28)
    let kernel_size = TensorShape(in_channels, in_channels, 5, 5)

    let res = MAXPOOL2D.forward[padding, stride](inputs, kernel_size)
    let torch_out = torch_maxpool2d(inputs, kernel_size, padding=padding, stride=stride)
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()


fn main():

    try:
        test_forward()
    except:
        print("Error")