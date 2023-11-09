from tensor import Tensor

from dainemo.utils.tensorutils import zero, fill, dot, tprint
from dainemo.utils.tensorutils import elwise_transform, elwise_pow, elwise_op, batch_tensor_elwise_op
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
    print(A)
    
    # <------------FILL------------>
    fill[dtype, nelts](A, 1.0)
    print(A)

    fill[dtype, nelts](B, 1.0)
    print(B)
    
    # <-------------DOT------------->
    let C = dot[dtype, nelts](A, B)
    print(C)


    # <-------------ELEMENT WISE TRANSFORM------------->
    var t0 = Tensor[dtype](2, 10)
    var result0 = Tensor[dtype](2, 10)
    fill[dtype, nelts](t0, 4)
    print(t0)

    result0 = elwise_transform[dtype, nelts, sqrt](t0)
    print(result0)
    
    result0 = elwise_transform[dtype, nelts, exp](t0)
    print(result0)

    result0 = elwise_transform[dtype, nelts, round](result0)
    print(result0)


    # <-------------ELEMENT WISE POW of TENSOR OPERATORS------------->
    var tpow = Tensor[dtype](1, 10)
    for i in range(10):
        tpow[i] = i
    print(tpow)

    let tpow_result = elwise_pow[dtype, nelts](tpow, 2)
    print(tpow_result)


    # <-------------ELEMENT WISE TENSOR-TENSOR OPERATORS------------->
    var t1 = Tensor[dtype](2, 10)
    var t2 = Tensor[dtype](2, 10)
    var result1 = Tensor[dtype](2, 10)
    fill[dtype, nelts](t1, 3.0)
    fill[dtype, nelts](t2, 3.0)
    print(t1)
    
    result1 = elwise_op[dtype, nelts, add](t1, t2)
    print(result1)

    result1 = elwise_op[dtype, nelts, sub](t1, t2)
    print(result1)

    result1 = elwise_op[dtype, nelts, mul](t1, t2)
    print(result1)

    result1 = elwise_op[dtype, nelts, div](t1, t2)
    print(result1)


    # <-------------ELEMENT WISE TESNOR-SCALAR OPERATORS------------->
    
    let a: SIMD[dtype, 1] = 2.0
    result1 = elwise_op[dtype, nelts, add](t1, a)
    print(result1)

    result1 = elwise_op[dtype, nelts, add](a, t1)
    print(result1)

    result1 = elwise_op[dtype, nelts, sub](t1, a)
    print(result1)

    result1 = elwise_op[dtype, nelts, mul](a, t1)
    print(result1)

    result1 = elwise_op[dtype, nelts, div](t1, a)
    print(result1)


    # <-------------ELEMENT WISE BATCH-TENSOR OPERATORS------------->
    var batch = Tensor[dtype](4, 10)
    for i in range(40):
        batch[i] = i
    print(batch)

    var t = Tensor[dtype](10)
    for i in range(10):
        t[i] = -i
    print(t)

    let batch_result1 = batch_tensor_elwise_op[dtype, nelts, add](batch, t)
    print(batch_result1)

    let batch_result2 = batch_tensor_elwise_op[dtype, nelts, sub](batch, t)
    print(batch_result2)

    let batch_result3 = batch_tensor_elwise_op[dtype, nelts, mul](batch, t)
    print(batch_result3)


    # <-------------ELEMENT WISE BATCH-BATCH OPERATORS------------->
    # Can be done similar to element wise TENSOR - TENSOR operators
    var batch1 = Tensor[dtype](4, 10)
    for i in range(40):
        batch1[i] = i
    print(batch1)
    
    var batch2 = Tensor[dtype](4, 10)
    for i in range(40):
        batch2[i] = i
    print(batch2)

    let batch_batch_result1 = elwise_op[dtype, nelts, add](batch1, batch2)
    print(batch_batch_result1)

    let batch_batch_result2 = elwise_op[dtype, nelts, sub](batch1, batch2)
    print(batch_batch_result2)

    let batch_batch_result3 = elwise_op[dtype, nelts, mul](batch1, batch2)
    print(batch_batch_result3)

    let batch_batch_result4 = elwise_pow[dtype, nelts](batch1, 2)
    print(batch_batch_result4)


    # <-------------SUM/MEAN/STD------------->
    for i in range(10):
        t[i] = i
    print(t)

    # Not specifying the axis takes all elements regardless of the shape
    let tensor_sum = tsum[dtype, nelts](t)
    print("sum: ", tensor_sum)

    let tensor_mean = tmean[dtype, nelts](t)
    print("mean: ", tensor_mean)

    let tensor_std = tstd[dtype, nelts](t)
    print("std: ", tensor_std)

    # When specifying the axis you can sum across batches
    fill[dtype, nelts](batch, 1)
    print(batch.shape().__str__())

    let batch_sum_0 = tsum[dtype, nelts](batch, axis=0)
    print(batch_sum_0)

    let batch_sum_1 = tsum[dtype, nelts](batch, axis=1)
    print(batch_sum_1)