from tensor import Tensor
from testing import assert_equal

from dainemo.autograd.ops.conv import get_result_shape


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


fn main():

    try:
        test_get_result_shape()
    except:
        print("Error")