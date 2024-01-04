from tensor import Tensor
from testing import assert_equal
from utils.index import Index

from dainemo.autograd.ops.conv import CONV2D
from dainemo.autograd.ops.conv import get_result_shape, pad, unpad
from dainemo.utils.tensorutils import fill


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


fn test_pad_unpad() raises:

    let inputs = Tensor[dtype](4, 28, 28)

    let padded = pad[2](inputs)
    # assert_equal(padded.shape()[0], 4)
    # assert_equal(padded.shape()[1], 30)
    # assert_equal(padded.shape()[2], 30)

    # var unpadded = unpad(padded)
    # assert_equal(unpadded.shape()[0], 4)
    # assert_equal(unpadded.shape()[1], 28)
    # assert_equal(unpadded.shape()[2], 28)






fn test_forward() raises:
    
    # padding=2, stride=1
    # input shape: (4, 28, 28)  kernel shape: (1, 16)
    # result_shape:  (32, 17)
    var inputs = Tensor[dtype](1, 28, 28)
    var kernel = Tensor[dtype](1 , 16)
    fill[dtype, nelts](inputs, 1.0)
    fill[dtype, nelts](kernel, 1.0)
    
    var bias = Tensor[dtype](99)

    let res = CONV2D.forward[2, 1](inputs, kernel, bias)

    print(res.tensor.shape())
    for i in range(res.tensor.shape()[1]):
        for j in range(res.tensor.shape()[2]):
            print_no_newline(res.tensor[Index(0, i, j)])
        print("")

    #######
    # Manually checked against pytorch conv2d
    #######
    # TODO: check equal output against pytorch 
    # After in_channels & out_channels are implemented
    """
    import torch
    from torch.nn.functional import conv2d
    import numpy as np

    inputs = torch.tensor(np.ones((1, 1, 28, 28)))
    weight = torch.tensor(np.ones((1, 1, 1, 16)))

    output = conv2d(inputs, weight, bias=None, stride=1, padding=2)

    print(output.numpy())
    """


    # padding=0, stride=1,
    # input shape: (4, 32, 17)  kernel shape: (2, 2)
    # result_shape:  (31, 16)
    inputs = Tensor[dtype](4, 32, 17)
    kernel = Tensor[dtype](2, 2)
    bias = Tensor[dtype](99)

    # let res2 = CONV2D.forward[0, 1](inputs, kernel, bias)





fn main():

    try:
        test_get_result_shape()
        test_pad_unpad()
        test_forward()
    except:
        print("Error")