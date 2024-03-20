from basalt import Tensor, TensorShape
from random import rand, randn
from algorithm import vectorize


@always_inline
fn rand_uniform[dtype: DType, shape: TensorShape](inout res: Tensor[dtype, shape], low: SIMD[dtype, 1], high: SIMD[dtype, 1]):
    rand[dtype](res.data(), res.num_elements()) # Uniform initialize the tensor between 0 and 1

    @parameter
    fn vecscale[nelts: Int](idx: Int):
        res.simd_store[nelts](idx, res.simd_load[nelts](idx) * (high - low) + low)

    vectorize[vecscale, nelts](res.num_elements())


@always_inline
fn rand_normal[dtype: DType, shape: TensorShape](inout res: Tensor[dtype, shape], mean: Float64, std: Float64):
    randn[dtype](res.data(), res.num_elements(), mean, std**2) # Normal distribution tensor initialization