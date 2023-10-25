from tensor import Tensor

from dainemo.utils.tensorutils import zero, fill, dot, tprint
from dainemo.utils.tensorutils import elwise_transform, elwise_op
from dainemo.utils.tensorutils import tsum, tmean, tstd

from math import sqrt, exp, round
from math import add, sub, mul, div


fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()

    var A: Tensor[dtype] = Tensor[dtype](2, 3)
    var B: Tensor[dtype] = Tensor[dtype](3, 2)
 
    # <------------ZERO------------>
    zero[dtype](A)
    tprint[dtype](A)
    
    # <------------FILL------------>
    fill[dtype, nelts](A, 1.0)
    tprint[dtype](A)

    fill[dtype, nelts](B, 1.0)
    tprint[dtype](B)
    
    # <-------------DOT------------->
    let C = dot[dtype, nelts](A, B)
    tprint[dtype](C)


    # <-------------ELEMENT WISE TRANSFORM------------->
    var t0 = Tensor[dtype](2, 10)
    var result0 = Tensor[dtype](2, 10)
    fill[dtype, nelts](t0, 4)
    tprint[dtype](t0)

    result0 = elwise_transform[dtype, nelts, sqrt](t0)
    tprint[dtype](result0)
    
    result0 = elwise_transform[dtype, nelts, exp](t0)
    tprint[dtype](result0)

    result0 = elwise_transform[dtype, nelts, round](result0)
    tprint[dtype](result0)


    # <-------------ELEMENT WISE TENSOR-TENSOR OPERATORS------------->
    var t1 = Tensor[dtype](2, 10)
    var t2 = Tensor[dtype](2, 10)
    var result1 = Tensor[dtype](2, 10)
    fill[dtype, nelts](t1, 3.0)
    fill[dtype, nelts](t2, 3.0)
    tprint[dtype](t1)
    
    result1 = elwise_op[dtype, nelts, add](t1, t2)
    tprint[dtype](result1)

    result1 = elwise_op[dtype, nelts, sub](t1, t2)
    tprint[dtype](result1)

    result1 = elwise_op[dtype, nelts, mul](t1, t2)
    tprint[dtype](result1)

    result1 = elwise_op[dtype, nelts, div](t1, t2)
    tprint[dtype](result1)


    # <-------------ELEMENT WISE TESNOR-SCALAR OPERATORS------------->
    
    let a: SIMD[dtype, 1] = 2.0
    result1 = elwise_op[dtype, nelts, add](t1, a)
    tprint[dtype](result1)

    result1 = elwise_op[dtype, nelts, add](a, t1)
    tprint[dtype](result1)

    result1 = elwise_op[dtype, nelts, sub](t1, a)
    tprint[dtype](result1)

    result1 = elwise_op[dtype, nelts, mul](a, t1)
    tprint[dtype](result1)

    result1 = elwise_op[dtype, nelts, div](t1, a)
    tprint[dtype](result1)


    # <-------------SUM/MEAN/STD------------->
    var t = Tensor[dtype](10)
    for i in range(10):
        t[i] = i
    tprint[dtype](t)

    let tensor_sum = tsum[dtype, nelts](t)
    print("sum: ", tensor_sum)

    let tensor_mean = tmean[dtype, nelts](t)
    print("mean: ", tensor_mean)

    let tensor_std = tstd[dtype, nelts](t)
    print("std: ", tensor_std)
