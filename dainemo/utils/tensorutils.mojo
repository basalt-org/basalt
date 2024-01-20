from tensor import Tensor, TensorShape
from utils.index import Index
from algorithm import vectorize, parallelize
from memory import memset_zero

from math import sqrt, pow, equal, max, min, abs, add, div


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


fn broadcast_elwise_op[
    dtype: DType,
    nelts: Int,
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
    let new_shape = broadcast_shapes(t1.shape(), t2.shape())
    var t_new = Tensor[dtype](new_shape)

    var strides1 = broadcast_calculate_strides(t1.shape(), t_new.shape())
    var strides2 = broadcast_calculate_strides(t2.shape(), t_new.shape())

    @parameter
    fn get_real_index(i: Int, shape: TensorShape, strides: DynamicVector[Int]) -> Int:
        var index_res = 0
        var linear_index = i
        for j in range(shape.rank() - 1, -1, -1):
            let stride = strides[j]
            let index = linear_index % shape[j]
            linear_index = linear_index // shape[j]
            index_res += index * stride

        return index_res

    @parameter
    fn vec_op[nelts: Int](i: Int):
        let index1 = get_real_index(i, t_new.shape(), strides1)
        let index2 = get_real_index(i, t_new.shape(), strides2)
        t_new.simd_store[nelts](
            i,
            func[dtype, nelts](
                t1.simd_load[nelts](index1), t2.simd_load[nelts](index2)
            ),
        )

    vectorize[1, vec_op](t_new.num_elements())

    _ = (strides1, strides2)

    return t_new


@always_inline
fn _reduce_sum[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    return x.reduce_add()


@always_inline
fn tsum[dtype: DType, nelts: Int](t: Tensor[dtype]) -> SIMD[dtype, 1]:
    let starting_value = 0
    return reduce[dtype, nelts, add, _reduce_sum](t, starting_value)


@always_inline
fn tsum[dtype: DType, nelts: Int](t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    let starting_value = 0
    return reduce[dtype, nelts, add, _reduce_sum](t, axis, starting_value)


@always_inline
fn tmean[dtype: DType, nelts: Int](t: Tensor[dtype]) -> SIMD[dtype, 1]:
    return tsum[dtype, nelts](t) / t.num_elements()


@always_inline
fn tmean[dtype: DType, nelts: Int](t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    let num_elements_axis: SIMD[dtype, 1] = t.dim(axis)
    return tsum[dtype, nelts](t, axis) / num_elements_axis
    

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


@always_inline
fn tstd[dtype: DType, nelts: Int](t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    let mu = tmean[dtype, nelts](t, axis)
    var variance = Tensor[dtype](mu.shape())
    let num_elements_axis: SIMD[dtype, 1] = t.dim(axis)
    
    let strides = calculate_strides(t.shape())
    let strides_mu = calculate_strides(mu.shape())

    @parameter
    fn get_t_index(i: Int, j: Int, axis: Int, shape: TensorShape, strides: DynamicVector[Int]) -> Int:
        var index_res = 0
        for k in range(shape.rank()):
            if k == axis:
                index_res += j * strides[k]
            else:
                index_res += (i % shape[k]) * strides[k]
        return index_res

    @parameter
    fn get_mu_index(i: Int, axis: Int, shape: TensorShape, strides: DynamicVector[Int]) -> Int:
        var index_res = 0
        for k in range(shape.rank()):
            if k != axis:
                index_res += (i % shape[k]) * strides[k]
        return index_res


    for i in range(t.num_elements() // t.dim(axis)):

        let mu_index = get_mu_index(i, axis, mu.shape(), strides_mu)
        
        @parameter
        fn vecvar[nelts: Int](j: Int):
            let t_index = get_t_index(i, j, axis, t.shape(), strides)
            let diff = t.simd_load[nelts](t_index) - mu[mu_index]
            variance[i] += (diff * diff).reduce_add()

        vectorize[nelts, vecvar](t.dim(axis))

        variance[i] /= num_elements_axis


    _ = (strides, strides_mu)
    return elwise_transform[dtype, nelts, sqrt](variance)


@always_inline
fn _reduce_max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    return x.reduce_max()


@always_inline
fn tmax[dtype: DType, nelts: Int](t: Tensor[dtype]) -> SIMD[dtype, 1]:
    let starting_value = math.limit.min_finite[dtype]()
    return reduce[dtype, nelts, max, _reduce_max](t, starting_value)


@always_inline
fn tmax[dtype: DType, nelts: Int](t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    let starting_value = math.limit.min_finite[dtype]()
    return reduce[dtype, nelts, max, _reduce_max](t, axis, starting_value)


@always_inline
fn reduce[
    dtype: DType,
    nelts: Int,
    op: fn[type: DType, simd_width: Int] (
        x: SIMD[type, simd_width], y: SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
    reduce_op: fn[type: DType, simd_width: Int] (x: SIMD[type, simd_width]) -> SIMD[
        type, 1
    ],
](t: Tensor[dtype], axis: Int, starting_value: SIMD[dtype, nelts]) -> Tensor[dtype]:
    var new_shape = DynamicVector[Int](t.rank())
    for i in range(t.rank()):
        if i == axis:
            new_shape.push_back(1)
        else:
            new_shape.push_back(t.dim(i))
    var t_new = Tensor[dtype](new_shape)

    let strides = calculate_strides(t.shape())

    @parameter
    fn parallel_reduce(i: Int):
        var m: SIMD[dtype, nelts] = starting_value

        let index_base = (i % strides[axis]) + (i // strides[axis]) * (
            strides[axis] * t.dim(axis)
        )

        @parameter
        fn axisreduce[_nelts: Int](j: Int):
            let index = index_base + j * strides[axis]
            if _nelts == 1:
                m[0] = op(m[0], t.simd_load[_nelts](index)[0])
            else:
                m = op(m, t.simd_load[nelts](index))

        vectorize[nelts, axisreduce](t.dim(axis))

        t_new[i] = reduce_op(m)

    parallelize[parallel_reduce](t.num_elements() // t.dim(axis))

    _ = strides
    return t_new


@always_inline
fn reduce[
    dtype: DType,
    nelts: Int,
    op: fn[type: DType, simd_width: Int] (
        x: SIMD[type, simd_width], y: SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
    reduce_op: fn[type: DType, simd_width: Int] (x: SIMD[type, simd_width]) -> SIMD[
        type, 1
    ],
](t: Tensor[dtype], starting_value: SIMD[dtype, nelts]) -> SIMD[dtype, 1]:
    var m: SIMD[dtype, nelts] = starting_value

    @parameter
    fn vecreduce[_nelts: Int](idx: Int):
        @parameter
        if _nelts == 1:
            m[0] = op(m[0], t.simd_load[_nelts](idx)[0])
        else:
            m = op(m, t.simd_load[nelts](idx))

    vectorize[nelts, vecreduce](t.num_elements())
    return reduce_op(m)


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
    # NOTE: This function could be deleted to use instead the transpose function
    var t_new = Tensor[dtype](t.dim(1), t.dim(0))

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
fn calculate_strides(shape: TensorShape) -> DynamicVector[Int]:
    var strides = DynamicVector[Int](shape.rank())
    strides.resize(shape.rank(), 1)

    for i in range(shape.rank() - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    return strides


@always_inline
fn broadcast_calculate_strides(
    shape: TensorShape, broadcast_shape: TensorShape
) -> DynamicVector[Int]:
    var strides = DynamicVector[Int](broadcast_shape.rank())
    strides.resize(broadcast_shape.rank(), 0)

    let diff = broadcast_shape.rank() - shape.rank()
    var stride = 1
    for i in range(shape.rank() - 1, -1, -1):
        if shape[i] != 1:
            strides[i + diff] = stride
            stride *= shape[i]

    return strides ^


@always_inline
fn transpose[
    dtype: DType, nelts: Int
](t: Tensor[dtype], dim_0: Int, dim_1: Int) -> Tensor[dtype]:
    """
    Create a new tensor transposing dim_0 and dim_1.
    """
    var axes = DynamicVector[Int](t.rank())

    for i in range(t.rank()):
        if i == dim_0:
            axes.push_back(dim_1)
        elif i == dim_1:
            axes.push_back(dim_0)
        else:
            axes.push_back(i)

    return transpose[dtype, nelts](t, axes)


@always_inline
fn transpose[dtype: DType, nelts: Int](t: Tensor[dtype]) -> Tensor[dtype]:
    """
    Create a new transposed tensor of the given tensor t.
    """
    var axes = DynamicVector[Int](t.rank())

    for i in range(t.rank() - 1, -1, -1):
        axes.push_back(i)

    return transpose[dtype, nelts](t, axes)


# It would be better to use VariadiList for axes, but because variadiclist can't be modified it wouldn't be possible to use overloaded transpose functions
@always_inline
fn transpose[
    dtype: DType, nelts: Int
](t: Tensor[dtype], axes: DynamicVector[Int]) -> Tensor[dtype]:
    """
    Create a new transposed tensor of the given tensor t.
    """
    # NOTE: The rank of of the t tensor should be 2 or more
    # NOTE: Axes should be the same size as the rank of t
    var new_shape = DynamicVector[Int](t.rank())
    for i in range(t.rank()):
        new_shape.push_back(t.dim(axes[i]))
    var t_new = Tensor[dtype](new_shape)

    let original_strides = calculate_strides(t.shape())
    let transposed_strides = calculate_strides(t_new.shape())

    @parameter
    fn p_transpose(i: Int):
        var new_index = 0
        var linear_index = i
        for j in range(t.rank()):
            let stride = original_strides[j]
            let index = linear_index // stride
            linear_index = linear_index % stride

            new_index += index * transposed_strides[axes[j]]

        t_new[new_index] = t[i]

        @parameter
        fn v_transpose[nelts: Int](j: Int):
            var new_index = 0
            let original_index = i * t.dim(t.rank() - 1) + j
            var linear_index = original_index
            for k in range(t.rank()):
                let stride = original_strides[k]
                let index = linear_index // stride
                linear_index = linear_index % stride

                new_index += index * transposed_strides[axes[k]]

            t_new.data().offset(new_index).simd_strided_store[nelts](
                t.simd_load[nelts](original_index),
                transposed_strides[axes[t.rank() - 1]],
            )

        vectorize[nelts, v_transpose](t.dim(t.rank() - 1))

    parallelize[p_transpose](t.num_elements() // t.dim(t.rank() - 1))

    _ = (original_strides, transposed_strides)

    return t_new


# TODO: Deprecate this function, as it is not used anymore
@always_inline
fn pad_zeros[
    dtype: DType, nelts: Int
](t: Tensor[dtype], pad_with: DynamicVector[Int]) -> Tensor[dtype]:
    """
    Pad a tensor with zeros along the specified axes of an N dimensional tensor.
    Number of values padded to the edges of each axis.
    Example: ((before_1, after_1), ... (before_N, after_N)).
    """

    # NOTE: The rank of of the t tensor should be equal to the size of pad_with devided by 2.
    # As pad_with contains (before, after) number of paddings for each axis.
    var new_shape = DynamicVector[Int](t.rank())
    for i in range(t.rank()):
        new_shape.push_back(t.dim(i) + pad_with[i * 2] + pad_with[i * 2 + 1])
    var t_new = Tensor[dtype](new_shape)

    let original_strides = calculate_strides(t.shape())
    let result_strides = calculate_strides(t_new.shape())

    # Parallelize over the first axis
    # NOTE: Possible dynamically choose the axis to parallelize over
    @parameter
    fn p_pad(i: Int):
        for j in range(t.num_elements() // t.dim(0)):
            let original_index = i * original_strides[0] + j

            # Padding contribution of the first dimention
            var dest_index = (i + pad_with[0]) * result_strides[0]

            # Calculate the contribution from each dimension
            var remaining_index = j % original_strides[0]
            for dim in range(1, t.rank()):
                let stride = original_strides[dim]
                let index = remaining_index // stride
                remaining_index = remaining_index % stride

                dest_index += (index + pad_with[dim * 2]) * result_strides[dim]

            # TODO: figure out vectorization
            t_new[dest_index] = t[original_index]

    parallelize[p_pad](t.dim(0))

    _ = (original_strides, result_strides)

    return t_new


@always_inline
fn broadcast_shapes(s1: TensorShape, s2: TensorShape) -> TensorShape:
    let ndim = max(s1.rank(), s2.rank())
    let diff = abs(s1.rank() - s2.rank())

    let big: TensorShape
    let small: TensorShape
    if s1.rank() > s2.rank():
        big = s1
        small = s2
    else:
        big = s2
        small = s1

    var res = DynamicVector[Int](ndim)
    res.resize(ndim, -1)

    for i in range(ndim - 1, diff - 1, -1):
        let a = big[i]
        let b = small[i - diff]
        if b == a:
            res[i] = a
        elif a == 1 or b == 1:
            res[i] = a * b
        else:
            # NOTE: consider assert and allow the function raises
            print("[ERROR] Shapes", s1, "and", s2, "cannot be broadcasted together.")
            return TensorShape(res)

    for i in range(diff - 1, -1, -1):
        res[i] = big[i]

    return TensorShape(res)


@always_inline
fn broadcast_shapes(*s: TensorShape) -> TensorShape:
    var result_shape = __get_address_as_lvalue(s[0])

    for i in range(1, len(s)):
        result_shape = broadcast_shapes(result_shape, __get_address_as_lvalue(s[i]))

    return result_shape
