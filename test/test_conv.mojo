from python.python import Python
from tensor import Tensor, TensorShape
from testing import assert_equal
from utils.index import Index

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
    if rank == 2:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1)))
    if rank == 3:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2)))
    if rank == 4:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3)))
    else:
        print("Error: rank not supported")
    
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


fn torch_conv2d(inputs: Tensor, kernel: Tensor, bias: PythonObject, padding: Int, stride: Int) raises -> Tensor[dtype]:
    try:
        let torch = Python.import_module("torch")
        let F = Python.import_module("torch.nn.functional")
        let np = Python.import_module("numpy")

        let expected = F.conv2d(
            torch.tensor(to_numpy(inputs)), 
            torch.tensor(to_numpy(kernel)),
            None,
            stride,
            padding
        )

        return to_tensor(expected.numpy())
    
    except:
        print("Error importing torch")
        return Tensor[dtype]()


fn test_forward() raises:
    
    # padding=2, stride=1
    # input shape: (4, 28, 28)  kernel shape: (1, 16)
    # result_shape:  (32, 17)
    alias padding_a = 2
    alias stride_a = 1
    var inputs = Tensor[dtype](1, 1, 28, 28)
    var kernel = Tensor[dtype](1, 1, 1 , 16)
    fill[dtype, nelts](inputs, 1.0)
    fill[dtype, nelts](kernel, 1.0)
    
    let bias = Tensor[dtype](99)
    let res = CONV2D.forward[padding_a, stride_a](inputs, kernel, bias)
    let expected = torch_conv2d(inputs, kernel, bias=None, padding=padding_a, stride=stride_a)

    print(to_numpy(res.tensor))
    print(to_numpy(expected))
    
    assert_tensors_equal(res.tensor, expected)
    GRAPH.reset()


    # padding=0, stride=1,
    # input shape: (4, 32, 17)  kernel shape: (2, 2)
    # result_shape:  (31, 16)
    alias padding_b = 0
    alias stride_b = 1
    inputs = Tensor[dtype](1, 1, 32, 17)
    kernel = Tensor[dtype](1, 1, 2, 2)
    fill[dtype, nelts](inputs, 1.0)
    fill[dtype, nelts](kernel, 1.0)

    let res_b = CONV2D.forward[padding_b, stride_b](inputs, kernel, bias)
    let expected_b = torch_conv2d(inputs, kernel, bias=None, padding=padding_b, stride=stride_b)

    print(to_numpy(res_b.tensor))
    print(to_numpy(expected_b))

    assert_tensors_equal(res_b.tensor, expected_b)
    GRAPH.reset()


fn main():

    try:
        test_get_result_shape()
        test_forward()
    except:
        print("Error")