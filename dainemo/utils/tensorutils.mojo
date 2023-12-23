from tensor import Tensor, TensorShape
from utils.index import Index
from algorithm import vectorize, parallelize
from memory import memset_zero

from math import sqrt, pow, equal, max, min


@always_inline
fn zero[dtype: DType](inout t: Tensor[dtype]):
    memset_zero[dtype](t.data(), t.num_elements())


@always_inline
fn fill[dtype: DType, nelts: Int](inout t: Tensor[dtype], val: SIMD[dtype, 1]):
    @parameter
    fn fill_vec[nelts: Int](idx: Int):
        t.simd_store[nelts](idx, t.simd_load[nelts](idx).splat(val))

    vectorize[nelts, fill_vec](t.num_elements())


@always_inline
fn elwise_transform[
    dtype: DType,
    nelts: Int,
    func: fn[dtype: DType, nelts: Int] (x: SIMD[dtype, nelts]) -> SIMD[dtype, nelts],
](t: Tensor[dtype]) -> Tensor[dtype]:
    var t_new = Tensor[dtype](t.shape())

    @parameter
    fn vecmath[nelts: Int](idx: Int):
        t_new.simd_store[nelts](idx, func[dtype, nelts](t.simd_load[nelts](idx)))

    vectorize[nelts, vecmath](t.num_elements())
    return t_new


@always_inline
fn elwise_pow[dtype: DType, nelts: Int](t: Tensor[dtype], x: Int) -> Tensor[dtype]:
    var t_new = Tensor[dtype](t.shape())

    @parameter
    fn vecpow[nelts: Int](idx: Int):
        t_new.simd_store[nelts](idx, pow(t.simd_load[nelts](idx), x))

    vectorize[nelts, vecpow](t.num_elements())
    return t_new


@always_inline
fn elwise_op[
    dtype: DType,
    nelts: Int,
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise operation on two tensors."""
    var t_new = Tensor[dtype](t1.shape())

    @parameter
    fn vecmath[nelts: Int](idx: Int):
        t_new.simd_store[nelts](
            idx, func[dtype, nelts](t1.simd_load[nelts](idx), t2.simd_load[nelts](idx))
        )

    vectorize[nelts, vecmath](t1.num_elements())
    return t_new


@always_inline
fn elwise_op[
    dtype: DType,
    nelts: Int,
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](t1: Tensor[dtype], a: SIMD[dtype, 1]) -> Tensor[dtype]:
    """Element-wise operation on a tensor and a scalar."""
    var t_new = Tensor[dtype](t1.shape())

    @parameter
    fn vecmath[nelts: Int](idx: Int):
        t_new.simd_store[nelts](idx, func[dtype, nelts](t1.simd_load[nelts](idx), a))

    vectorize[nelts, vecmath](t1.num_elements())
    return t_new


@always_inline
fn elwise_op[
    dtype: DType,
    nelts: Int,
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](a: SIMD[dtype, 1], t1: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise operation on a tensor and a scalar."""
    var t_new = Tensor[dtype](t1.shape())

    @parameter
    fn vecmath[nelts: Int](idx: Int):
        t_new.simd_store[nelts](idx, func[dtype, nelts](a, t1.simd_load[nelts](idx)))

    vectorize[nelts, vecmath](t1.num_elements())
    return t_new


@always_inline
fn batch_tensor_elwise_op[
    dtype: DType,
    nelts: Int,
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](t_batch: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise operation on between a batch of tensors t_batch and a tensor t2."""
    var t_new = Tensor[dtype](t_batch.shape())

    @parameter
    fn row_op(r: Int):
        @parameter
        fn vecmath[nelts: Int](c: Int):
            t_new.simd_store[nelts](
                r * t_batch.dim(1) + c,
                func[dtype, nelts](
                    t_batch.simd_load[nelts](r * t_batch.dim(1) + c),
                    t2.simd_load[nelts](c),
                ),
            )

        vectorize[nelts, vecmath](t_batch.dim(1))

    parallelize[row_op](t_batch.dim(0), t_batch.dim(0))
    return t_new


@always_inline
fn tsum[dtype: DType, nelts: Int](t: Tensor[dtype]) -> SIMD[dtype, 1]:
    var s: SIMD[dtype, 1] = 0

    @parameter
    fn vecsum[nelts: Int](idx: Int):
        s += t.simd_load[nelts](idx).reduce_add()

    vectorize[nelts, vecsum](t.num_elements())
    return s


# from testing import assert_equal
@always_inline
fn tsum[dtype: DType, nelts: Int](t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    let d: Int = 1 if axis == 0 else 0
    let t_new = Tensor[dtype](1, t.dim(d)) if axis == 0 else Tensor[dtype](t.dim(d), 1)

    @parameter
    fn parallel_sum(i: Int):
        var s: SIMD[dtype, 1] = 0

        @parameter
        fn axissum[nelts: Int](j: Int):
            let index = j * t.dim(d) + i if axis == 0 else i * t.dim(axis) + j
            s += t.simd_load[nelts](index).reduce_add()

        vectorize[nelts, axissum](t.dim(axis))
        t_new[i] = s

    parallelize[parallel_sum](t.dim(d), t.dim(d))

    return t_new


@always_inline
fn tmean[dtype: DType, nelts: Int](t: Tensor[dtype]) -> SIMD[dtype, 1]:
    return tsum[dtype, nelts](t) / t.num_elements()


@always_inline
fn tstd[dtype: DType, nelts: Int](t: Tensor[dtype]) -> SIMD[dtype, 1]:
    var mu: SIMD[dtype, 1] = tmean[dtype, nelts](t)
    var variance: SIMD[dtype, 1] = 0

    @parameter
    fn vecvar[nelts: Int](idx: Int):
        let diff = t.simd_load[nelts](idx) - mu
        variance += (diff * diff).reduce_add()

    vectorize[nelts, vecvar](t.num_elements())

    return sqrt(variance / t.num_elements())


fn tmean2[dtype: DType](t: Tensor[dtype], axis: Int = 0):
    """Calculate mean of a 2D tensor along a specified axis."""
    # TODO: every mean of vector can be calulated in parallel where each mean calculation can be vectorized
    pass


fn tstd2[dtype: DType](t: Tensor[dtype], axis: Int = 0):
    """Calculate standard deviation of a 2D tensor along a specified axis."""
    # TODO
    pass


@always_inline
fn dot[dtype: DType, nelts: Int](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    var C = Tensor[dtype](A.dim(0), B.dim(1))
    memset_zero[dtype](C.data(), C.num_elements())

    @parameter
    fn calc_row(m: Int):
        for k in range(
            B.dim(0)
        ):  # TODO: test dot(4x1x28x28, 784x32) = (4x32) // mnist case

            @parameter
            fn dot[nelts: Int](n: Int):
                C.simd_store[nelts](
                    m * C.dim(1) + n,
                    C.simd_load[nelts](m * C.dim(1) + n)
                    + A[m, k] * B.simd_load[nelts](k * B.dim(1) + n),
                )

            vectorize[nelts, dot](C.dim(1))

    parallelize[calc_row](C.dim(0), C.dim(0))

    return C


@always_inline
fn transpose_2D[dtype: DType, nelts: Int](t: Tensor[dtype]) -> Tensor[dtype]:
    var t_new = Tensor[dtype](t.dim(1), t.dim(0))

    # TODO: figure out vectorization
    # TODO: make it work for any rank

    let stride = t.dim(0)

    @parameter
    fn proc_row(i: Int):
        @parameter
        fn proc_column[nelts: Int](j: Int):
            t_new.data().offset(j * t.dim(0) + i).simd_strided_store[nelts](
                t.simd_load[nelts](i * t.dim(1) + j), stride
            )

        vectorize[nelts, proc_column](t.dim(1))

    parallelize[proc_row](t.dim(0))

    return t_new


@always_inline
fn transpose[
    dtype: DType, nelts: Int
](t: Tensor[dtype], dim_0: Int, dim_1: Int) -> Tensor[dtype]:
    """
    Create a new transposed tensor of the given tensor t.
    """

    # Get the new shape of the transposed tensor
    var new_shape = DynamicVector[Int](t.rank())

    for i in range(t.rank()):
        if i == dim_0:
            new_shape.push_back(t.dim(dim_1))
        elif i == dim_1:
            new_shape.push_back(t.dim(dim_0))
        else:
            new_shape.push_back(t.dim(i))

    var t_new = Tensor[dtype](new_shape)

    # Get the strides of the old and new tensors
    var dims = StaticIntTuple[2]()
    dims[0] = min(dim_0, dim_1)  # last dimension (reading from right to left)
    dims[1] = max(dim_0, dim_1)  # first dimension (reading from right to left)
    # example: tensor(2x3), strides (6, 3, 1)
    var strides_old = DynamicVector[Int](t.rank() + 1)
    var strides_new = DynamicVector[Int](t_new.rank() + 1)
    strides_old.resize(t.rank() + 1, 1)
    strides_new.resize(t_new.rank() + 1, 1)

    for i in range(t.rank() - 1, -1, -1):
        strides_old[i] = strides_old[i + 1] * t.dim(i)
        strides_new[i] = strides_new[i + 1] * t_new.dim(i)

    # Transpose the tensor
    let i_range = strides_old[0] // strides_old[dims[0]]
    let j_range = t.dim(dims[0])
    let k_range = strides_old[dims[0] + 1] // strides_old[dims[1]]
    let l_range = t.dim(dims[1])
    let m_range = strides_new[dims[1] + 1]

    # NOTE: The reason why we use strides_old_shape and strides_new_shape is 
    # because it seems there is a *bug* when using dynamic vectors inside a 
    # parameter function? or a parameter function that is used in parallelized. 
    # If we use the dynamic vector inside the parallelized function, the memory 
    # of the dynamic vector is not initialized.
    var strides_old_shape = TensorShape(strides_old)
    var strides_new_shape = TensorShape(strides_new)

    @parameter
    fn p_transpose(i: Int):
        for j in range(j_range):
            for k in range(k_range):
                for l in range(l_range):
                    for m in range(m_range):
                        let index_old =
                            i * strides_old_shape[dims[0]]
                            + j * strides_old_shape[dims[0] + 1]
                            + k * strides_old_shape[dims[1]]
                            + l * strides_old_shape[dims[1] + 1]
                            + m
                        let index_new =
                            i * strides_new_shape[dims[0]]
                            + l * strides_new_shape[dims[0] + 1]
                            + k * strides_new_shape[dims[1]]
                            + j * strides_new_shape[dims[1] + 1]
                            + m

                        t_new[index_new] = t[index_old]
    
    parallelize[p_transpose](i_range)

    return t_new
