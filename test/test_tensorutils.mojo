from tensor import TensorShape
from random import rand
from testing import assert_equal, assert_true, assert_almost_equal

from dainemo import dtype, nelts
from dainemo.utils.tensorutils import zero, fill, dot
from dainemo.utils.tensorutils import (
    elwise_transform,
    elwise_pow,
    elwise_op,
    broadcast_shapes,
    broadcast_elwise_op,
    get_reduce_shape,
)
from dainemo.utils.tensorutils import tsum, tmean, tstd, tmax #, transpose_2D, transpose, pad_zeros

from math import sqrt, exp, round
from math import add, sub, mul, div



fn assert_tensors_equal(t1: Tensor[dtype], t2: Tensor[dtype], mode: String = "exact") raises:
    # Assert equal shapes
    assert_equal(t1.num_elements(), t2.num_elements())
    assert_equal(t1.rank(), t2.rank())
    for i in range(t1.rank()):
        assert_equal(t1.dim(i), t2.dim(i))
    # Assert equal values
    for i in range(t1.num_elements()):
        if mode == "exact":
            assert_equal(t1[i], t2[i])
        elif mode == "almost":
            assert_almost_equal[dtype, 1](t1[i], t2[i], rtol=1e-5)
        else:
            print("Mode must be 'exact' or 'almost'")


# <------------ZERO------------>
fn test_zero() raises:
    var A = Tensor[dtype](2, 3)
    var B = rand[dtype](2, 3)
    zero[dtype](B)
    assert_tensors_equal(A, B)


# <------------FILL------------>
fn test_fill() raises:
    var A = Tensor[dtype](2, 3)
    var B = Tensor[dtype](2, 3)
    for i in range(A.num_elements()):
        A[i] = 1.0
    fill[dtype, nelts](B, 1.0)
    assert_tensors_equal(A, B)


# <-------------DOT------------->
fn test_dot() raises:
    alias a_shape = TensorShape(2, 3)
    alias b_shape = TensorShape(3, 2)
    var A = Tensor[dtype](a_shape)
    var B = Tensor[dtype](b_shape)
    fill[dtype, nelts](A, 1.0)
    fill[dtype, nelts](B, 1.0)

    var C = Tensor[dtype](2, 2)
    dot[a_shape, b_shape](C, A, B)
    var C_expected = Tensor[dtype](2, 2)
    fill[dtype, nelts](C_expected, 3.0)
    assert_tensors_equal(C, C_expected)

    var D = Tensor[dtype](3, 3)
    dot[b_shape, a_shape](D, B, A)
    var D_expected = Tensor[dtype](3, 3)
    fill[dtype, nelts](D_expected, 2.0)
    assert_tensors_equal(D, D_expected)


# <-------------ELEMENT WISE TRANSFORM------------->
fn test_elwise_transform() raises:
    var A = Tensor[dtype](2, 10)
    var B = Tensor[dtype](2, 10)
    var C = Tensor[dtype](2, 10)
    var D = Tensor[dtype](2, 10)
    fill[dtype, nelts](A, 4)
    fill[dtype, nelts](B, 2)
    fill[dtype, nelts](C, exp[dtype, 1](2))
    fill[dtype, nelts](D, 7)

    var A_res = Tensor[dtype](2, 10)
    elwise_transform[sqrt](A_res, A)
    assert_tensors_equal(A_res, B)

    var B_res = Tensor[dtype](2, 10)
    elwise_transform[exp](B_res, B)
    assert_tensors_equal(B_res, C)

    var C_res = Tensor[dtype](2, 10)
    elwise_transform[round](C_res, C)
    assert_tensors_equal(C_res, D)


# <-------------ELEMENT WISE POW of TENSOR OPERATORS------------->
fn test_elwise_pow() raises:
    var A = Tensor[dtype](1, 10)
    var B = Tensor[dtype](1, 10)
    for i in range(10):
        A[i] = i
        B[i] = i**2

    var A_res = Tensor[dtype](1, 10)
    elwise_pow(A_res, A, 2)
    assert_tensors_equal(A_res, B)


# <-------------ELEMENT WISE TENSOR-TENSOR OPERATORS------------->
fn test_elwise_tensor_tensor() raises:
    alias t1_shape = TensorShape(2, 10)
    alias t2_shape = TensorShape(2, 10)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    fill[dtype, nelts](t1, 3.0)
    fill[dtype, nelts](t2, 3.0)

    var result1 = Tensor[dtype](2, 10)
    elwise_op[t1_shape, t2_shape, add](result1, t1, t2)
    var result1_expected = Tensor[dtype](2, 10)
    fill[dtype, nelts](result1_expected, 6.0)
    assert_tensors_equal(result1, result1_expected)

    var result2 = Tensor[dtype](2, 10)
    elwise_op[t1_shape, t2_shape, sub](result2, t1, t2)
    var result2_expected = Tensor[dtype](2, 10)
    assert_tensors_equal(result2, result2_expected)

    var result3 = Tensor[dtype](2, 10)
    elwise_op[t1_shape, t2_shape, mul](result3, t1, t2)
    var result3_expected = Tensor[dtype](2, 10)
    fill[dtype, nelts](result3_expected, 9.0)
    assert_tensors_equal(result3, result3_expected)

    var result4 = Tensor[dtype](2, 10)
    elwise_op[t1_shape, t2_shape, div](result4, t1, t2)
    var result4_expected = Tensor[dtype](2, 10)
    fill[dtype, nelts](result4_expected, 1.0)
    assert_tensors_equal(result4, result4_expected)


# <-------------ELEMENT WISE TESNOR-SCALAR OPERATORS------------->
fn test_elwise_tensor_scalar() raises:
    var a: SIMD[dtype, 1] = 2.0
    var t1 = Tensor[dtype](2, 10)
    fill[dtype, nelts](t1, 1.0)
    var result = Tensor[dtype](2, 10)

    elwise_op[add](result, t1, a)
    var result1_expected = Tensor[dtype](2, 10)
    fill[dtype, nelts](result1_expected, 3.0)
    assert_tensors_equal(result, result1_expected)

    elwise_op[add](result, a, t1)
    assert_tensors_equal(result, result1_expected)

    elwise_op[sub](result, t1, a)
    var result3_expected = Tensor[dtype](2, 10)
    fill[dtype, nelts](result3_expected, -1)
    assert_tensors_equal(result, result3_expected)

    elwise_op[mul](result, a, t1)
    var result4_expected = Tensor[dtype](2, 10)
    fill[dtype, nelts](result4_expected, 2)
    assert_tensors_equal(result, result4_expected)

    elwise_op[div](result, t1, a)
    var result5_expected = Tensor[dtype](2, 10)
    fill[dtype, nelts](result5_expected, 0.5)
    assert_tensors_equal(result, result5_expected)


# <-------------ELEMENT WISE STRIDE ITER OPERATORS------------->
fn test_elwise_broadcast_tensor() raises:
    alias t1_shape = TensorShape(2, 3, 4)
    alias t2_shape = TensorShape(5, 2, 1, 4)
    alias res_shape = broadcast_shapes(t1_shape, t2_shape)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)

    fill[dtype, nelts](t1, 3.0)
    for i in range(40):
        t2[i] = i + 1

    var result1 = Tensor[dtype](res_shape)
    elwise_op[t1_shape, t2_shape, add](result1, t1, t2)
    var result1_expected = Tensor[dtype](5, 2, 3, 4)
    # fill expected tensor
    for i in range(40):
        for j in range(3):
            var index = (i % 4) + ((i // 4) * 12) + j * 4
            result1_expected[index] = 3.0 + (i + 1)
    assert_tensors_equal(result1, result1_expected)


# <-------------SUM/MEAN/STD------------->
from test_tensorutils_data import SumMeanStdData

fn test_sum_mean_std() raises:
    var t = Tensor[dtype](2, 10)
    var s = 0
    for i in range(20):
        t[i] = i + 1
        s += i + 1

    # Not specifying the axis takes all elements regardless of the shape
    var tensor_sum = tsum(t)
    assert_equal(tensor_sum, s)

    var tensor_mean = tmean(t)
    assert_equal(tensor_mean, s / 20)

    var tensor_std = tstd(t)
    var expected_std: SIMD[dtype, 1] = 0
    for i in range(20):
        expected_std += (i + 1 - tensor_mean) ** 2
    expected_std = sqrt(expected_std / 20)
    assert_equal(tensor_std, expected_std)

    # When specifying the axis you can sum across batches
    # Axis 0
    var batch_sum_0 = Tensor[dtype](get_reduce_shape(t.shape(), axis=0))
    tsum(batch_sum_0, t, axis=0)
    var expected_batch_sum_0 = Tensor[dtype](1, 10)
    for i in range(10):
        expected_batch_sum_0[i] = (i + 1) + (i + 1 + 10)
    assert_tensors_equal(batch_sum_0, expected_batch_sum_0)

    var batch_mean_0 = Tensor[dtype](get_reduce_shape(t.shape(), axis=0))
    tmean(batch_mean_0, t, axis=0)
    var expected_batch_mean_0 = Tensor[dtype](1, 10)
    for i in range(10):
        expected_batch_mean_0[i] = expected_batch_sum_0[i] / 2
    assert_tensors_equal(batch_mean_0, expected_batch_mean_0)

    var batch_std_0 = Tensor[dtype](get_reduce_shape(t.shape(), axis=0))
    tstd(batch_std_0, t, axis=0)
    var expected_batch_std_0 = Tensor[dtype](1, 10)
    fill[dtype, nelts](expected_batch_std_0, 5)
    assert_tensors_equal(batch_std_0, expected_batch_std_0)

    # Axis 1
    var batch_sum_1 = Tensor[dtype](get_reduce_shape(t.shape(), axis=1))
    tsum(batch_sum_1, t, axis=1)
    var expected_batch_sum_1 = Tensor[dtype](2, 1)
    expected_batch_sum_1[0] = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
    expected_batch_sum_1[1] = 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20
    assert_tensors_equal(batch_sum_1, expected_batch_sum_1)

    var batch_mean_1 = Tensor[dtype](get_reduce_shape(t.shape(), axis=1))
    tmean(batch_mean_1, t, axis=1)
    var expected_batch_mean_1 = Tensor[dtype](2, 1)
    expected_batch_mean_1[0] = expected_batch_sum_1[0] / 10
    expected_batch_mean_1[1] = expected_batch_sum_1[1] / 10
    assert_tensors_equal(batch_mean_1, expected_batch_mean_1)

    var batch_std_1 = Tensor[dtype](get_reduce_shape(t.shape(), axis=1))
    tstd(batch_std_1, t, axis=1)
    var expected_batch_std_1 = Tensor[dtype](2, 1)
    fill[dtype, nelts](expected_batch_std_1, 2.8722813129425049)
    assert_tensors_equal(batch_std_1, expected_batch_std_1)

fn test_sum_mean_std_n() raises:
    var t = Tensor[dtype](3, 4, 5)
    var s = 0
    for i in range(60):
        t[i] = i + 1
        s += i + 1

    # Not specifying the axis takes all elements regardless of the shape
    var tensor_sum = tsum(t)
    assert_equal(tensor_sum, s)

    var tensor_mean = tmean(t)
    assert_equal(tensor_mean, s / 60)

    var tensor_std = tstd(t)
    var expected_std: SIMD[dtype, 1] = 0
    for i in range(60):
        expected_std += (i + 1 - tensor_mean) ** 2
    expected_std = sqrt(expected_std / 60)
    assert_equal(tensor_std, expected_std)

    # When specifying the axis you can sum across batches
    # Axis 0
    var data = SumMeanStdData.generate_3d_axis_0()
    var batch_sum_0 = Tensor[dtype](get_reduce_shape(t.shape(), axis=0))
    tsum(batch_sum_0, t, axis=0)
    assert_tensors_equal(batch_sum_0, data.expected_sum)

    var batch_mean_0 = Tensor[dtype](get_reduce_shape(t.shape(), axis=0))
    tmean(batch_mean_0, t, axis=0)
    assert_tensors_equal(batch_mean_0, data.expected_mean)

    var batch_std_0 = Tensor[dtype](get_reduce_shape(t.shape(), axis=0))
    tstd(batch_std_0, t, axis=0)
    assert_tensors_equal(batch_std_0, data.expected_std)

    # When specifying the axis you can sum across batches
    # Axis 1
    data = SumMeanStdData.generate_3d_axis_1()
    var batch_sum_1 = Tensor[dtype](get_reduce_shape(t.shape(), axis=1))
    tsum(batch_sum_1, t, axis=1)
    assert_tensors_equal(batch_sum_1, data.expected_sum)

    var batch_mean_1 = Tensor[dtype](get_reduce_shape(t.shape(), axis=1))
    tmean(batch_mean_1, t, axis=1)
    assert_tensors_equal(batch_mean_1, data.expected_mean)

    var batch_std_1 = Tensor[dtype](get_reduce_shape(t.shape(), axis=1))
    tstd(batch_std_1, t, axis=1)
    assert_tensors_equal(batch_std_1, data.expected_std)

    # When specifying the axis you can sum across batches
    # Axis 2
    data = SumMeanStdData.generate_3d_axis_2()
    var batch_sum_2 = Tensor[dtype](get_reduce_shape(t.shape(), axis=2))
    tsum(batch_sum_2, t, axis=2)
    assert_tensors_equal(batch_sum_2, data.expected_sum)

    var batch_mean_2 = Tensor[dtype](get_reduce_shape(t.shape(), axis=2))
    tmean(batch_mean_2, t, axis=2)
    assert_tensors_equal(batch_mean_2, data.expected_mean)

    var batch_std_2 = Tensor[dtype](get_reduce_shape(t.shape(), axis=2))
    tstd(batch_std_2, t, axis=2)
    assert_tensors_equal(batch_std_2, data.expected_std)


# <-------------MAX------------->
fn test_max() raises:
    var t = Tensor[dtype](2, 3, 2)
    for i in range(12):
        t[i] = i + 1

    var tensor_max = tmax(t)
    assert_equal(tensor_max, 12)

    @parameter
    fn fill_tensor[size: Int](inout tensor: Tensor[dtype], values: StaticIntTuple[size]):
        for i in range(tensor.num_elements()):
            tensor[i] = values[i]

    var tensor_max_axis_0 = Tensor[dtype](get_reduce_shape(t.shape(), axis=0))
    tmax(tensor_max_axis_0, t, axis=0)
    var expected_max_axis_0_temp = StaticIntTuple[6](7, 8, 9, 10, 11, 12)
    var expected_max_axis_0 = Tensor[dtype](1, 3, 2)
    fill_tensor(expected_max_axis_0, expected_max_axis_0_temp)
    assert_tensors_equal(tensor_max_axis_0, expected_max_axis_0)

    var tensor_max_axis_1 = Tensor[dtype](get_reduce_shape(t.shape(), axis=1))
    tmax(tensor_max_axis_1, t, axis=1)
    var expected_max_axis_1_temp = StaticIntTuple[4](5, 6, 11, 12)
    var expected_max_axis_1 = Tensor[dtype](2, 1, 2)
    fill_tensor(expected_max_axis_1, expected_max_axis_1_temp)
    assert_tensors_equal(tensor_max_axis_1, expected_max_axis_1)

    var tensor_max_axis_2 = Tensor[dtype](get_reduce_shape(t.shape(), axis=2))
    tmax(tensor_max_axis_2, t, axis=2)
    var expected_max_axis_2_temp = StaticIntTuple[6](2, 4, 6, 8, 10, 12)
    var expected_max_axis_2 = Tensor[dtype](2, 3, 1)
    fill_tensor(expected_max_axis_2, expected_max_axis_2_temp)
    assert_tensors_equal(tensor_max_axis_2, expected_max_axis_2)


# # <-------------TRANSPOSE------------->
# from test_tensorutils_data import TransposeData


# fn test_transpose() raises:
#     # Transpose 2D
#     var data = TransposeData.generate_1_2dim_test_case()
#     var transposed = transpose_2D[dtype, nelts](data.A)
#     assert_tensors_equal(transposed, data.expected)

#     # Transpose 2 dimensions
#     data = TransposeData.generate_2_2dim_test_case()
#     transposed = transpose[dtype, nelts](
#         data.A, data.transpose_dims[0], data.transpose_dims[1]
#     )
#     assert_tensors_equal(transposed, data.expected)

#     data = TransposeData.generate_3_2dim_test_case()
#     transposed = transpose[dtype, nelts](
#         data.A, data.transpose_dims[0], data.transpose_dims[1]
#     )
#     assert_tensors_equal(transposed, data.expected)

#     data = TransposeData.generate_4_2dim_test_case()
#     transposed = transpose[dtype, nelts](
#         data.A, data.transpose_dims[0], data.transpose_dims[1]
#     )
#     assert_tensors_equal(transposed, data.expected)

#     # Transpose all dimensions
#     data = TransposeData.generate_1_alldim_test_case()
#     var transpose_dims = DynamicVector[Int]()
#     for i in range(len(data.transpose_dims)):
#         transpose_dims.push_back(data.transpose_dims[i])

#     transposed = transpose[dtype, nelts](data.A, transpose_dims)
#     assert_tensors_equal(transposed, data.expected)

#     # Transpose (reverse)
#     data = TransposeData.generate_1_transpose_test_case()
#     transposed = transpose[dtype, nelts](data.A)
#     assert_tensors_equal(transposed, data.expected)


# # <-------------FLATTEN/RESHAPE------------->
# fn test_flatten() raises:
#     var A = Tensor[dtype](2, 3)
#     var B = Tensor[dtype](6)
#     for i in range(6):
#         A[i] = i + 1
#         B[i] = i + 1

#     var A_flat = A.reshape(
#         TensorShape(A.num_elements())
#     )  # or A.ireshape to modify in place
#     assert_tensors_equal(A_flat, B)

#     var A_resh = A_flat.reshape(A.shape())  # or A_flat.ireshape to modify in place
#     assert_tensors_equal(A_resh, A)


# # <-------------PADDING------------->
# from test_tensorutils_data import PaddingData

# fn test_padding() raises:
#     # 1D padding (only after)
#     var data = PaddingData.generate_1d_test_case_after()
#     var padded_data = pad_zeros[dtype, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 1D padding (before and after)
#     data = PaddingData.generate_1d_test_case_before_after()
#     padded_data = pad_zeros[dtype, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 2D padding
#     data = PaddingData.generate_2d_test_case()
#     padded_data = pad_zeros[dtype, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 3D padding (simple)
#     data = PaddingData.generate_3d_test_case_simple()
#     padded_data = pad_zeros[dtype, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

#     # 3D padding
#     data = PaddingData.generate_3d_test_case()
#     padded_data = pad_zeros[dtype, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)
    
#     # 4D padding
#     data = PaddingData.generate_4d_test_case()
#     padded_data = pad_zeros[dtype, nelts](data.A, data.pad_with)
#     assert_tensors_equal(padded_data, data.expected)

    

fn main():
    try:
        test_zero()
        test_fill()
        test_dot()
        test_elwise_transform()
        test_elwise_pow()
        test_elwise_tensor_tensor()
        test_elwise_tensor_scalar()
        test_elwise_broadcast_tensor()
        test_sum_mean_std()
        test_sum_mean_std_n()
        test_max()
        # test_transpose()
        # test_flatten()
        # test_padding()
    except:
        print("[ERROR] Error in tensorutils.py")
