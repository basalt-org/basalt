from python.python import Python
from tensor import Tensor, TensorShape
from testing import assert_equal
from utils.index import Index
from random import rand

from dainemo import GRAPH
from dainemo.autograd.ops.conv import CONV2D
from dainemo.autograd.ops.conv import get_result_shape
from dainemo.utils.tensorutils import fill
from test_tensorutils import assert_tensors_equal


alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


fn test_get_result_shape() raises:

    # padding=2, stride=1
    # input shape: (4, 28, 28)  kernel shape: (1, 16)
    # result:  (32, 17)
    var inputs = Tensor[dtype](4, 28, 28)
    var kernel = Tensor[dtype](1, 16)

    var res = get_result_shape[2, 1](inputs.shape(), kernel.shape())
    assert_equal(res[0], 32)
    assert_equal(res[1], 17)

    # padding=0, stride=1,
    # input shape: (4, 32, 17)  kernel shape: (2, 2)
    # result:  (31, 16)
    inputs = Tensor[dtype](4, 32, 17)
    kernel = Tensor[dtype](2, 2)

    res = get_result_shape[0, 1](inputs.shape(), kernel.shape())
    assert_equal(res[0], 31)
    assert_equal(res[1], 16)


def to_numpy(tensor: Tensor) -> PythonObject:
    let np = Python.import_module("numpy")

    rank = tensor.rank()
    var pyarray: PythonObject = np.array([0])
    if rank == 1:
        pyarray = np.empty((tensor.dim(0)))
    elif rank == 2:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1)))
    elif rank == 3:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2)))
    elif rank == 4:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3)))
    else:
        print("Error: rank not supported: ", rank)
    
    for i in range(tensor.num_elements()):
        pyarray.itemset((i), tensor[i])
   
    return pyarray


fn to_tensor(np_array: PythonObject) raises-> Tensor[dtype]:
    var shape = DynamicVector[Int]()
    for i in range(np_array.ndim):
        shape.push_back(np_array.shape[i].to_float64().to_int())
    
    var tensor = Tensor[dtype](TensorShape(shape))

    for i in range(tensor.num_elements()):
        tensor[i] = np_array.ravel()[i].to_float64().cast[dtype]()

    return tensor


@value
struct torch_conv2d_output:
    var expected: Tensor[dtype]
    var expected_inputs_grad: Tensor[dtype]
    var expected_kernel_grad: Tensor[dtype]
    var expected_bias_grad: Tensor[dtype]


fn torch_conv2d(inputs: Tensor, kernel: Tensor, bias: Tensor, padding: Int, stride: Int, upper_grad: Tensor) -> torch_conv2d_output:
    let out: torch_conv2d_output
    
    try:
        let torch = Python.import_module("torch")
        let F = Python.import_module("torch.nn.functional")
        let np = Python.import_module("numpy")

        let inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)
        let weights = torch.from_numpy(to_numpy(kernel)).requires_grad_(True)
        let bias = torch.from_numpy(to_numpy(bias)).requires_grad_(True)
        
        let expected = F.conv2d(
            inputs, 
            weights,
            bias,
            stride,
            padding
        )

        # uppergrad & backwards
        let upper_grad = torch.from_numpy(to_numpy(upper_grad))
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
        let d = Tensor[dtype](1)
        let out: torch_conv2d_output = torch_conv2d_output(d, d, d, d)
        return out


fn test_forward_1() raises:
    # padding=2, stride=1
    # input shape: (4, 1, 28, 28)  kernel shape: (1, 1, 1, 16)
    # result_shape:  (4, 1, 32, 17)
    alias padding = 2
    alias stride = 1
    var inputs = Tensor[dtype](4, 1, 28, 28)
    var kernel = Tensor[dtype](1, 1, 1 , 16)
    fill[dtype, nelts](inputs, 1.0)
    fill[dtype, nelts](kernel, 1.0)
    let bias = Tensor[dtype](1)

    let res = CONV2D.forward[padding, stride](inputs, kernel, bias)
    let torch_out = torch_conv2d(inputs, kernel, bias=bias, padding=padding, stride=stride, upper_grad=res.grad)
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()


fn test_forward_2() raises:
    # padding=0, stride=1,
    # input shape: (4, 1, 32, 17)  kernel shape: (1, 1, 2, 2)
    # result_shape:  (4, 1, 31, 16)
    alias padding = 0
    alias stride = 1
    var inputs = Tensor[dtype](4, 1, 32, 17)
    var kernel = Tensor[dtype](1, 1, 2, 2)
    fill[dtype, nelts](inputs, 1.0)
    fill[dtype, nelts](kernel, 1.0)
    let bias = Tensor[dtype](1)

    let res = CONV2D.forward[padding, stride](inputs, kernel, bias)
    let torch_out = torch_conv2d(inputs, kernel, bias=bias, padding=padding, stride=stride, upper_grad=res.grad)
    assert_tensors_equal(res.tensor, torch_out.expected)
    GRAPH.reset_all()


fn test_backward_1() raises:
    # padding=2, stride=1
    alias padding = 2
    alias stride = 1
    alias batch = 4
    alias in_channels = 2
    alias out_channels = 3
    var inputs = Tensor[dtype](batch, in_channels, 4, 4)
    var kernel = Tensor[dtype](out_channels, in_channels, 2 , 2)
    fill[dtype, nelts](inputs, 1.0)
    fill[dtype, nelts](kernel, 1.0)
    let bias = Tensor[dtype](out_channels)

    let res = CONV2D.forward[padding, stride](inputs, kernel, bias)

    let gn = GRAPH.graph[GRAPH.get_node_idx(res.uuid)]
    assert_equal(gn.parents.size, 3)
    var upper_grad: Tensor[dtype] = rand[dtype](res.tensor.shape())

    let ug1 = gn.backward_fn(upper_grad, gn.parents, 0) # inputs.grad
    let ug2 = gn.backward_fn(upper_grad, gn.parents, 1) # kernel.grad
    let ug3 = gn.backward_fn(upper_grad, gn.parents, 2) # bias.grad

    let torch_out = torch_conv2d(inputs, kernel, bias=bias, padding=padding, stride=stride, upper_grad=upper_grad)
    assert_tensors_equal(res.tensor, torch_out.expected)
    print(to_numpy(ug1))
    print("----")
    print(to_numpy(torch_out.expected_inputs_grad))
    # assert_tensors_equal(ug1, torch_out.expected_inputs_grad) # TODO
    # assert_tensors_equal(ug2, torch_out.expected_kernel_grad) # TODO
    print(to_numpy(ug3))
    print("----")
    print(to_numpy(torch_out.expected_bias_grad))
    # assert_tensors_equal(ug3, torch_out.expected_bias_grad) # TODO: assert_tensors_ALMOST_equal
    GRAPH.reset_all()


fn main():

    try:
        test_get_result_shape()
        test_forward_1()
        test_forward_2()
        test_backward_1()
    except:
        print("Error")