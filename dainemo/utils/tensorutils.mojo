from algorithm import vectorize, parallelize
from memory import memset_zero
from tensor import TensorShape
from math import sqrt, pow, equal, max, min, abs, add, div
from random import rand



@always_inline
fn zero[dtype: DType](inout t: Tensor[dtype]):
    memset_zero[dtype](t.data(), t.num_elements())


@always_inline
fn fill[dtype: DType, nelts: Int](inout t: Tensor[dtype], val: SIMD[dtype, 1]):
    @parameter
    fn fill_vec[nelts: Int](idx: Int):
        t.simd_store[nelts](idx, t.simd_load[nelts](idx).splat(val))

    vectorize[fill_vec, nelts](t.num_elements())


# ----- Functions to access positions in tensor data -----
fn get_real_index[
    broadcast_shape: TensorShape
](i: Int, strides_shape: DynamicVector[Int]) -> Int:
    # broadcast_shape is of same rank as strides_shape (the not broadcasted shape), because of broadcast_calculate_strides
    var index_res = 0
    var linear_index = i

    @parameter
    fn unroll_dims[dim: Int]():
        var j = (broadcast_shape.rank() - 1) - dim

        var stride = strides_shape[j]
        var index = linear_index % broadcast_shape[j]
        linear_index = linear_index // broadcast_shape[j]
        index_res += index * stride

    unroll[unroll_dims, broadcast_shape.rank()]()

    return index_res



# ----- Broadcast functions -----
@always_inline
fn broadcast_shapes(s1: TensorShape, s2: TensorShape) -> TensorShape:
    var ndim = max(s1.rank(), s2.rank())
    var diff = abs(s1.rank() - s2.rank())

    var big: TensorShape
    var small: TensorShape
    if s1.rank() > s2.rank():
        big = s1
        small = s2
    else:
        big = s2
        small = s1

    var res = DynamicVector[Int]()
    res.resize(ndim, -1)

    for i in range(ndim - 1, diff - 1, -1):
        var a = big[i]
        var b = small[i - diff]
        if b == a:
            res[i] = a
        elif a == 1 or b == 1:
            res[i] = a * b
        else:
            # NOTE: consider assert and allow the function raises
            var message: String = "[ERROR] Shapes " + str(s1) + " and " + str(
                s2
            ) + " cannot be broadcasted together."
            print(message)
            # raise Error(message)

    for i in range(diff - 1, -1, -1):
        res[i] = big[i]

    return TensorShape(res)


@always_inline
fn broadcast_shapes(*s: TensorShape) -> TensorShape:
    var result_shape = s[0]

    for i in range(1, len(s)):
        result_shape = broadcast_shapes(result_shape, s[i])

    return result_shape


@always_inline
fn broadcast_calculate_strides(
    shape: TensorShape, broadcast_shape: TensorShape
) -> DynamicVector[Int]:
    var strides = DynamicVector[Int]()
    strides.resize(broadcast_shape.rank(), 0)

    var diff = broadcast_shape.rank() - shape.rank()
    var stride = 1
    for i in range(shape.rank() - 1, -1, -1):
        if shape[i] != 1:
            strides[i + diff] = stride
            stride *= shape[i]

    return strides ^


@always_inline
fn calculate_strides(shape: TensorShape) -> DynamicVector[Int]:
    var strides = DynamicVector[Int]()
    strides.resize(shape.rank(), 1)

    for i in range(shape.rank() - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    return strides


# ----- Dot functions -----
@always_inline
fn dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    alias res1 = t2_shape[1]
    memset_zero[dtype](res.data(), res.num_elements())

    @parameter
    fn calc_row(m: Int):
        for k in range(t2_shape[0]):

            @parameter
            fn dot[nelts: Int](n: Int):
                res.simd_store[nelts](
                    m * res1 + n,
                    res.simd_load[nelts](m * res1 + n)
                    + t1[m, k] * t2.simd_load[nelts](k * res1 + n),
                )

            vectorize[dot, nelts](res1)

    parallelize[calc_row](t1_shape[0], t1_shape[0])


fn dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](inout C: Tensor[dtype], A: Tensor[dtype], B: Tensor[dtype]):
    memset_zero[dtype](C.data(), C.num_elements())

    @parameter
    fn calc_row(i: Int):
        for j in range(B_shape[0]):

            @parameter
            fn calc_row_A_B[nelts: Int](k: Int):
                var A_pos = i * A.dim(1) + k
                var B_pos = j * A.dim(1) + k
                var t_new_pos = i * C.dim(1) + j

                C[t_new_pos] += (
                    A.simd_load[nelts](A_pos) * B.simd_load[nelts](B_pos)
                ).reduce_add()

            vectorize[calc_row_A_B, nelts, A_shape[1]]()

    parallelize[calc_row](A_shape[0], 1)


fn dot_transpose_t1[
    A_shape: TensorShape, B_shape: TensorShape
](inout C: Tensor[dtype], A: Tensor[dtype], B: Tensor[dtype]):
    memset_zero[dtype](C.data(), C.num_elements())

    @parameter
    fn calc_row(i: Int):
        for j in range(A_shape[0]):

            @parameter
            fn calc_row_t_new_B[nelts: Int](k: Int):
                var A_pos = j * A.dim(1) + i
                var B_pos = j * B.dim(1) + k
                var t_new_pos = i * C.dim(1) + k

                C.simd_store[nelts](
                    t_new_pos,
                    C.simd_load[nelts](t_new_pos)
                    + A[A_pos] * B.simd_load[nelts](B_pos),
                )

            vectorize[calc_row_t_new_B, nelts, B_shape[1]]()

    parallelize[calc_row](A_shape[1], 1)


# ----- Element-wise unary operations -----
@always_inline
fn elwise_transform[
    func: fn[dtype: DType, nelts: Int] (x: SIMD[dtype, nelts]) -> SIMD[dtype, nelts],
](inout res: Tensor[dtype], t: Tensor[dtype]):
    @parameter
    fn vecmath[nelts: Int](idx: Int):
        res.simd_store[nelts](idx, func[dtype, nelts](t.simd_load[nelts](idx)))

    vectorize[vecmath, nelts](t.num_elements())


# ----- Element-wise binary operations -----
@always_inline
fn elwise_pow(inout res: Tensor[dtype], t: Tensor[dtype], x: Int):
    @parameter
    fn vecpow[nelts: Int](idx: Int):
        res.simd_store[nelts](idx, pow(t.simd_load[nelts](idx), x))

    vectorize[vecpow, nelts](t.num_elements())


@always_inline
fn elwise_op[
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    alias broadcast: Bool = (t1_shape != t2_shape)
    alias is_scalar: Bool = (t2_shape == TensorShape(1))

    @parameter
    if t2_shape == TensorShape(1):
        elwise_op[func](res, t1, t2[0])
    elif t1_shape == TensorShape(1):
        elwise_op[func](res, t1[0], t2)
    elif broadcast and not is_scalar:
        alias res_shape = broadcast_shapes(t1_shape, t2_shape)
        broadcast_elwise_op[t1_shape, t2_shape, res_shape, func](res, t1, t2)
    else:
        elwise_op[func](res, t1, t2)


@always_inline
fn elwise_op[
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    """Element-wise operation on two tensors of equal shape."""

    @parameter
    fn vecmath[nelts: Int](idx: Int):
        res.simd_store[nelts](
            idx, func[dtype, nelts](t1.simd_load[nelts](idx), t2.simd_load[nelts](idx))
        )

    vectorize[vecmath, nelts](t1.num_elements())


@always_inline
fn elwise_op[
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](inout res: Tensor[dtype], t1: Tensor[dtype], a: SIMD[dtype, 1]):
    """Element-wise operation on a tensor and a scalar."""

    @parameter
    fn vecmath[nelts: Int](idx: Int):
        res.simd_store[nelts](idx, func[dtype, nelts](t1.simd_load[nelts](idx), a))

    vectorize[vecmath, nelts](t1.num_elements())

@always_inline
fn elwise_op[
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](inout res: Tensor[dtype], a: SIMD[dtype, 1], t1: Tensor[dtype]):
    """Element-wise operation on a tensor and a scalar."""

    @parameter
    fn vecmath[nelts: Int](idx: Int):
        res.simd_store[nelts](idx, func[dtype, nelts](a, t1.simd_load[nelts](idx)))
    
    vectorize[vecmath, nelts](t1.num_elements())


fn broadcast_elwise_op[
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    res_shape: TensorShape,
    func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    alias strides1 = broadcast_calculate_strides(t1_shape, res_shape)
    alias strides2 = broadcast_calculate_strides(t2_shape, res_shape)

    @parameter
    fn vec_op[nelts: Int](i: Int):
        var index1 = get_real_index[res_shape](i, strides1)
        var index2 = get_real_index[res_shape](i, strides2)

        res.simd_store[nelts](
            i,
            func[dtype, nelts](
                t1.simd_load[nelts](index1), t2.simd_load[nelts](index2)
            ),
        )

    vectorize[vec_op, 1](res.num_elements())


fn unbroadcast_add[
    unbroadcast_res_shape: TensorShape, original_shape: TensorShape
](inout unbroadcast_res: Tensor[dtype], original: Tensor[dtype]):
    # original_shape is broadcast shape
    @parameter
    if original_shape == unbroadcast_res_shape:
        elwise_op[add](unbroadcast_res, unbroadcast_res, original)
    elif original_shape == TensorShape(1):
        elwise_op[add](unbroadcast_res, unbroadcast_res, original[0])
    else:
        alias strides_unbroadcast_res = broadcast_calculate_strides(
            unbroadcast_res_shape, original_shape
        )

        @parameter
        fn vec_op[nelts: Int](i: Int):
            var index = get_real_index[original_shape](i, strides_unbroadcast_res)
            unbroadcast_res[index] += original.simd_load[nelts](i).reduce_add()

        vectorize[vec_op, nelts](original.num_elements())


# ---- Transform functions -----
@always_inline
fn transpose_2D[dtype: DType, nelts: Int](t: Tensor[dtype]) -> Tensor[dtype]:
    var t_new = Tensor[dtype](t.dim(1), t.dim(0))

    var stride = t.dim(0)

    @parameter
    fn proc_row(i: Int):
        @parameter
        fn proc_column[nelts: Int](j: Int):
            t_new.data().offset(j * t.dim(0) + i).simd_strided_store[nelts](
                t.simd_load[nelts](i * t.dim(1) + j), stride
            )

        vectorize[proc_column, nelts](t.dim(1))

    parallelize[proc_row](t.dim(0))

    return t_new ^


# ----- Reduction functions -----
@always_inline
fn reduce[
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

    vectorize[vecreduce, nelts](t.num_elements())
    return reduce_op(m)


@always_inline
fn reduce[
    op: fn[type: DType, simd_width: Int] (
        x: SIMD[type, simd_width], y: SIMD[type, simd_width]
    ) -> SIMD[type, simd_width],
    reduce_op: fn[type: DType, simd_width: Int] (x: SIMD[type, simd_width]) -> SIMD[
        type, 1
    ],
](t: Tensor[dtype], axis: Int, starting_value: SIMD[dtype, nelts]) -> Tensor[dtype]:
    var new_shape = DynamicVector[Int](capacity=t.rank())
    for i in range(t.rank()):
        if i == axis:
            new_shape.push_back(1)
        else:
            new_shape.push_back(t.dim(i))
    var t_new = Tensor[dtype](new_shape)

    var strides = calculate_strides(t.shape())

    @parameter
    fn parallel_reduce(i: Int):
        var m: SIMD[dtype, nelts] = starting_value

        var index_base = (i % strides[axis]) + (i // strides[axis]) * (
            strides[axis] * t.dim(axis)
        )

        @parameter
        fn axisreduce[_nelts: Int](j: Int):
            var index = index_base + j * strides[axis]
            if _nelts == 1:
                m[0] = op(m[0], t.simd_load[_nelts](index)[0])
            else:
                m = op(m, t.simd_load[nelts](index))

        vectorize[axisreduce, nelts](t.dim(axis))

        t_new[i] = reduce_op(m)

    parallelize[parallel_reduce](t.num_elements() // t.dim(axis))

    _ = strides
    return t_new ^


@always_inline
fn _reduce_sum[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    return x.reduce_add()


@always_inline
fn tsum(t: Tensor[dtype]) -> SIMD[dtype, 1]:
    var starting_value = 0
    return reduce[add, _reduce_sum](t, starting_value)


@always_inline
fn tmean(t: Tensor[dtype]) -> SIMD[dtype, 1]:
    return tsum(t) / t.num_elements()


@always_inline
fn tstd(t: Tensor[dtype]) -> SIMD[dtype, 1]:
    var mu: SIMD[dtype, 1] = tmean(t)
    var variance: SIMD[dtype, 1] = 0

    @parameter
    fn vecvar[nelts: Int](idx: Int):
        var diff = t.simd_load[nelts](idx) - mu
        variance += (diff * diff).reduce_add()

    vectorize[vecvar, nelts](t.num_elements())

    return sqrt(variance / t.num_elements())


@always_inline
fn tsum(t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    var starting_value = 0
    return reduce[add, _reduce_sum](t, axis, starting_value)


@always_inline
fn tmean(t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    var num_elements_axis: SIMD[dtype, 1] = t.dim(axis)
    return tsum(t, axis) / num_elements_axis
    

@always_inline
fn tstd(t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    var mu = tmean(t, axis)
    var variance = Tensor[dtype](mu.shape())
    var num_elements_axis: SIMD[dtype, 1] = t.dim(axis)
    
    var strides = calculate_strides(t.shape())
    var strides_mu = calculate_strides(mu.shape())

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

        var mu_index = get_mu_index(i, axis, mu.shape(), strides_mu)
        
        @parameter
        fn vecvar[nelts: Int](j: Int):
            var t_index = get_t_index(i, j, axis, t.shape(), strides)
            var diff = t.simd_load[nelts](t_index) - mu[mu_index]
            variance[i] += (diff * diff).reduce_add()

        vectorize[vecvar, nelts](t.dim(axis))

        variance[i] /= num_elements_axis


    _ = (strides, strides_mu)
    elwise_transform[sqrt](variance, variance)
    return variance ^


@always_inline
fn _reduce_max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, 1]:
    return x.reduce_max()


@always_inline
fn tmax(t: Tensor[dtype]) -> SIMD[dtype, 1]:
    var starting_value = math.limit.min_finite[dtype]()
    return reduce[max, _reduce_max](t, starting_value)


@always_inline
fn tmax(t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
    var starting_value = math.limit.min_finite[dtype]()
    return reduce[max, _reduce_max](t, axis, starting_value)



# @always_inline
# fn transpose[
#     dtype: DType, nelts: Int
# ](t: Tensor[dtype], dim_0: Int, dim_1: Int) -> Tensor[dtype]:
#     """
#     Create a new tensor transposing dim_0 and dim_1.
#     """
#     var axes = DynamicVector[Int](t.rank())

#     for i in range(t.rank()):
#         if i == dim_0:
#             axes.push_back(dim_1)
#         elif i == dim_1:
#             axes.push_back(dim_0)
#         else:
#             axes.push_back(i)

#     return transpose[dtype, nelts](t, axes)


# @always_inline
# fn transpose[dtype: DType, nelts: Int](t: Tensor[dtype]) -> Tensor[dtype]:
#     """
#     Create a new transposed tensor of the given tensor t.
#     """
#     var axes = DynamicVector[Int](t.rank())

#     for i in range(t.rank() - 1, -1, -1):
#         axes.push_back(i)

#     return transpose[dtype, nelts](t, axes)


# # It would be better to use VariadiList for axes, but because variadiclist can't be modified it wouldn't be possible to use overloaded transpose functions
# @always_inline
# fn transpose[
#     dtype: DType, nelts: Int
# ](t: Tensor[dtype], axes: DynamicVector[Int]) -> Tensor[dtype]:
#     """
#     Create a new transposed tensor of the given tensor t.
#     """
#     # NOTE: The rank of of the t tensor should be 2 or more
#     # NOTE: Axes should be the same size as the rank of t
#     var new_shape = DynamicVector[Int](t.rank())
#     for i in range(t.rank()):
#         new_shape.push_back(t.dim(axes[i]))
#     var t_new = Tensor[dtype](new_shape)

#     var original_strides = calculate_strides(t.shape())
#     var transposed_strides = calculate_strides(t_new.shape())

#     @parameter
#     fn p_transpose(i: Int):
#         var new_index = 0
#         var linear_index = i
#         for j in range(t.rank()):
#             var stride = original_strides[j]
#             var index = linear_index // stride
#             linear_index = linear_index % stride

#             new_index += index * transposed_strides[axes[j]]

#         t_new[new_index] = t[i]

#         @parameter
#         fn v_transpose[nelts: Int](j: Int):
#             var new_index = 0
#             var original_index = i * t.dim(t.rank() - 1) + j
#             var linear_index = original_index
#             for k in range(t.rank()):
#                 var stride = original_strides[k]
#                 var index = linear_index // stride
#                 linear_index = linear_index % stride

#                 new_index += index * transposed_strides[axes[k]]

#             t_new.data().offset(new_index).simd_strided_store[nelts](
#                 t.simd_load[nelts](original_index),
#                 transposed_strides[axes[t.rank() - 1]],
#             )

#         vectorize[nelts, v_transpose](t.dim(t.rank() - 1))

#     parallelize[p_transpose](t.num_elements() // t.dim(t.rank() - 1))

#     _ = (original_strides, transposed_strides)

#     return t_new


# # TODO: Deprecate this function, as it is not used anymore
# @always_inline
# fn pad_zeros[
#     dtype: DType, nelts: Int
# ](t: Tensor[dtype], pad_with: DynamicVector[Int]) -> Tensor[dtype]:
#     """
#     Pad a tensor with zeros along the specified axes of an N dimensional tensor.
#     Number of values padded to the edges of each axis.
#     Example: ((before_1, after_1), ... (before_N, after_N)).
#     """

#     # NOTE: The rank of of the t tensor should be equal to the size of pad_with devided by 2.
#     # As pad_with contains (before, after) number of paddings for each axis.
#     var new_shape = DynamicVector[Int](t.rank())
#     for i in range(t.rank()):
#         new_shape.push_back(t.dim(i) + pad_with[i * 2] + pad_with[i * 2 + 1])
#     var t_new = Tensor[dtype](new_shape)

#     var original_strides = calculate_strides(t.shape())
#     var result_strides = calculate_strides(t_new.shape())

#     # Parallelize over the first axis
#     # NOTE: Possible dynamically choose the axis to parallelize over
#     @parameter
#     fn p_pad(i: Int):
#         for j in range(t.num_elements() // t.dim(0)):
#             var original_index = i * original_strides[0] + j

#             # Padding contribution of the first dimention
#             var dest_index = (i + pad_with[0]) * result_strides[0]

#             # Calculate the contribution from each dimension
#             var remaining_index = j % original_strides[0]
#             for dim in range(1, t.rank()):
#                 var stride = original_strides[dim]
#                 var index = remaining_index // stride
#                 remaining_index = remaining_index % stride

#                 dest_index += (index + pad_with[dim * 2]) * result_strides[dim]

#             # TODO: figure out vectorization
#             t_new[dest_index] = t[original_index]

#     parallelize[p_pad](t.dim(0))

#     _ = (original_strides, result_strides)

#     return t_new