from random import rand
from algorithm import vectorize


@always_inline
fn rand_uniform(inout res: Tensor[dtype], low: SIMD[dtype, 1], high: SIMD[dtype, 1]):
    rand[dtype](res.data(), res.num_elements()) # Uniform initialize the tensor between 0 and 1

    @parameter
    fn vecscale[nelts: Int](idx: Int):
        res.simd_store[nelts](idx, res.simd_load[nelts](idx) * (high - low) + low)

    vectorize[vecscale, nelts](res.num_elements())