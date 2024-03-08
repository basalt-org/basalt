from math import add
from tensor import TensorShape
from dainemo.utils.tensorutils import fill, elwise_op

alias dtype = DType.float32
alias nelts = simdwidthof[dtype]()

fn generate_tensor(*shape: Int) -> Tensor[dtype]:
    var A = Tensor[dtype](shape)
    var size = A.num_elements()
    for i in range(size):
        A[i] = i + 1
    return A ^

fn generate_expected_tensor[
    size: Int
](data: StaticIntTuple[size], *shape: Int) -> Tensor[dtype]:
    var A = Tensor[dtype](shape)
    for i in range(size):
        A[i] = data[i]
    return A ^



struct TransposeData:
    var A: Tensor[dtype]
    var expected: Tensor[dtype]
    var transpose_dims: VariadicList[Int]

    fn __init__(inout self, A: Tensor[dtype], expected: Tensor[dtype], transpose_dims: VariadicList[Int]):
        self.A = A
        self.expected = expected
        self.transpose_dims = transpose_dims

    @staticmethod
    fn generate_1_2dim_test_case() -> TransposeData:
        var A = generate_tensor(2, 3)
        var expected = StaticIntTuple[6](1, 4, 2, 5, 3, 6)
        var tranpose_dims = VariadicList[Int](0, 1)
        var B = generate_expected_tensor(expected, 3, 2)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_2_2dim_test_case() -> TransposeData:
        var A = generate_tensor(2, 3, 2)
        var expected = StaticIntTuple[12](1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12)
        var tranpose_dims = VariadicList[Int](0, 2)
        var B = generate_expected_tensor(expected, 2, 3, 2)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_3_2dim_test_case() -> TransposeData:
        var A = generate_tensor(2, 3, 2, 3)
        var expected = StaticIntTuple[36](1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 
        10, 11, 12, 16, 17, 18, 19, 20, 21, 25, 26, 27, 31, 32, 33, 22, 23, 24, 
        28, 29, 30, 34, 35, 36)
        var tranpose_dims = VariadicList[Int](1, 2)
        var B = generate_expected_tensor(expected, 2, 2, 3, 3)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_4_2dim_test_case() -> TransposeData:
        var A = generate_tensor(3, 2, 3, 2, 3)
        var expected = StaticIntTuple[108](1, 2, 3, 19, 20, 21, 7, 8, 9, 25, 
        26, 27, 13, 14, 15, 31, 32, 33, 4, 5, 6, 22, 23, 24, 10, 11, 12, 28, 
        29, 30, 16, 17, 18, 34, 35, 36, 37, 38, 39, 55, 56, 57, 43, 44, 45, 61, 
        62, 63, 49, 50, 51, 67, 68, 69, 40, 41, 42, 58, 59, 60, 46, 47, 48, 64, 
        65, 66, 52, 53, 54, 70, 71, 72, 73, 74, 75, 91, 92, 93, 79, 80, 81, 97, 
        98, 99, 85, 86, 87, 103, 104, 105, 76, 77, 78, 94, 95, 96, 82, 83, 84, 
        100, 101, 102, 88, 89, 90, 106, 107, 108
        )
        var tranpose_dims = VariadicList[Int](1, 3)
        var B = generate_expected_tensor(expected, 3, 2, 3, 2, 3)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_1_alldim_test_case() -> TransposeData:
        var A = generate_tensor(2, 3, 2, 3)
        var expected = StaticIntTuple[36](1,4,2,5,3,6,19,22,20,23,21,24,7,10,8,
        11,9,12,25,28,26,29,27,30,13,16,14,17,15,18,31,34,32,35,33,36)
        var tranpose_dims = VariadicList[Int](1, 0, 3, 2)
        var B = generate_expected_tensor(expected, 3, 2, 3, 2)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_1_transpose_test_case() -> TransposeData:
        var A = generate_tensor(2, 3, 2, 3)
        var expected = StaticIntTuple[36](1,19,7,25,13,31,4,22,10,28,16,34,2,20,
        8,26,14,32,5,23,11,29,17,35,3,21,9,27,15,33,6,24,12,30,18,36)
        var tranpose_dims = VariadicList[Int](3, 2, 1, 0)
        var B = generate_expected_tensor(expected, 3, 2, 3, 2)

        return TransposeData(A, B, tranpose_dims)


struct PaddingData:
    var A: Tensor[dtype]
    var expected: Tensor[dtype]
    var pad_with: DynamicVector[Int]

    fn __init__(inout self, A: Tensor[dtype], expected: Tensor[dtype], pad_with: DynamicVector[Int]):
        self.A = A
        self.expected = expected
        self.pad_with = pad_with

    @staticmethod
    fn generate_1d_test_case_after() -> PaddingData:
        var A = generate_tensor(2)

        var expected = StaticIntTuple[4](1, 2, 0, 0)
        var pad_with = DynamicVector[Int]()
        pad_with.push_back(0) # before
        pad_with.push_back(2) # after

        var B = generate_expected_tensor(expected, 4)

        return PaddingData(A, B, pad_with)

    @staticmethod
    fn generate_1d_test_case_before_after() -> PaddingData:
        var A = generate_tensor(3)

        var expected = StaticIntTuple[6](0, 0, 1, 2, 3, 0)
        var pad_with = DynamicVector[Int]()
        pad_with.push_back(2) # before
        pad_with.push_back(1) # after

        var B = generate_expected_tensor(expected, 6)

        return PaddingData(A, B, pad_with)

    @staticmethod
    fn generate_2d_test_case() -> PaddingData:
        var A = generate_tensor(2, 2)

        var expected = StaticTuple[45](
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 2, 0, 0, 0, 0,
            0, 0, 0, 3, 4, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        var pad_with = DynamicVector[Int]()
        pad_with.push_back(1) # before_1
        pad_with.push_back(2) # after_1
        pad_with.push_back(3) # before_2
        pad_with.push_back(4) # after_2

        var B = generate_expected_tensor[45](expected, 5, 9)

        return PaddingData(A, B, pad_with)

    @staticmethod
    fn generate_3d_test_case_simple() -> PaddingData:
        var A = generate_tensor(2, 2, 2)

        var expected = StaticIntTuple[16](
            0, 0, 1, 2, 3, 4, 0, 0,
            0, 0, 5, 6, 7, 8, 0, 0
        )
        var pad_with = DynamicVector[Int]()
        pad_with.push_back(0) # before_1
        pad_with.push_back(0) # after_1
        pad_with.push_back(1) # before_2
        pad_with.push_back(1) # after_2
        pad_with.push_back(0) # before_3
        pad_with.push_back(0) # after_3

        var B = generate_expected_tensor[16](expected, 2, 4, 2)

        return PaddingData(A, B, pad_with)

    @staticmethod
    fn generate_3d_test_case() -> PaddingData:
        var A = generate_tensor(1, 2, 3)

        var expected = StaticIntTuple[45](
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 2, 3, 0, 0,
            4, 5, 6, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        )
        var pad_with = DynamicVector[Int]()
        pad_with.push_back(1) # before_1
        pad_with.push_back(1) # after_1
        pad_with.push_back(1) # before_2
        pad_with.push_back(0) # after_2
        pad_with.push_back(0) # before_3
        pad_with.push_back(2) # after_3

        var B = generate_expected_tensor[45](expected, 3, 3, 5)

        return PaddingData(A, B, pad_with)

    
    @staticmethod
    fn generate_4d_test_case() -> PaddingData:
        var A = generate_tensor(2, 2, 2, 2)

        var expected = StaticIntTuple[81](
            1,  2,  0,  3,  4,  0,  0,  0,  0,
            5,  6,  0,  7,  8,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,
            9,  10, 0,  11, 12, 0,  0,  0,  0,
            13, 14, 0,  15, 16, 0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0
        )

        var pad_with = DynamicVector[Int]()
        pad_with.push_back(0) # before_1
        pad_with.push_back(1) # after_1
        pad_with.push_back(0) # before_2
        pad_with.push_back(1) # after_2
        pad_with.push_back(0) # before_3
        pad_with.push_back(1) # after_3
        pad_with.push_back(0) # before_4
        pad_with.push_back(1) # after_4

        var B = generate_expected_tensor[81](expected, 3, 3, 3, 3)

        return PaddingData(A, B, pad_with)



struct SumMeanStdData:
    var A: Tensor[dtype]
    var axis: Int
    var expected_sum: Tensor[dtype]
    var expected_mean: Tensor[dtype]
    var expected_std: Tensor[dtype]
    

    fn __init__(inout self, A: Tensor[dtype], axis: Int, expected_sum: Tensor[dtype], expected_mean: Tensor[dtype], expected_std: Tensor[dtype]):
        self.A = A
        self.axis = axis
        self.expected_sum = expected_sum
        self.expected_mean = expected_mean
        self.expected_std = expected_std
        
    @staticmethod
    fn generate_3d_axis_0() -> SumMeanStdData:
        var A = generate_tensor(3, 4, 5)
        var axis = 0

        var expected_sum = StaticIntTuple[20](
            63,  66,  69,  72,  75,
            78,  81,  84,  87,  90,
            93,  96,  99,  102, 105,
            108, 111, 114, 117, 120,
        )

        var expected_mean = StaticIntTuple[20](
            21, 22, 23, 24, 25,
            26, 27, 28, 29, 30,
            31, 32, 33, 34, 35,
            36, 37, 38, 39, 40,
        )
        
        var expected_std = Tensor[dtype](1, 4, 5)
        fill[dtype, nelts](expected_std, 16.32993162)
        
        var B = generate_expected_tensor[20](expected_sum, 1, 4, 5)
        var C = generate_expected_tensor[20](expected_mean, 1, 4, 5)

        return SumMeanStdData(A, axis, B, C, expected_std)

    @staticmethod
    fn generate_3d_axis_1() -> SumMeanStdData:
        var A = generate_tensor(3, 4, 5)
        var axis = 1

        var expected_sum = StaticIntTuple[15](
            34,  38,  42,  46,  50,
            114, 118, 122, 126, 130,
            194, 198, 202, 206, 210,
        )

        var expected_mean = StaticIntTuple[15](
            8,  9,  10, 11, 12,
            28, 29, 30, 31, 32,
            48, 49, 50, 51, 52,
        ) # 0.5 added afterwards 

        var expected_std = Tensor[dtype](3, 1, 5)
        fill[dtype, nelts](expected_std, 5.59016994)

        var B = generate_expected_tensor[15](expected_sum, 3, 1, 5)
        var C = generate_expected_tensor[15](expected_mean, 3, 1, 5)
        elwise_op[add](C, C, 0.5)

        return SumMeanStdData(A, axis, B, C, expected_std)

    @staticmethod
    fn generate_3d_axis_2() -> SumMeanStdData:
        var A = generate_tensor(3, 4, 5)
        var axis = 2

        var expected_sum = StaticIntTuple[12](
            15,  40,  65,  90,
            115, 140, 165, 190,
            215, 240, 265, 290,
        )

        var expected_mean = StaticIntTuple[12](
            3,  8,  13, 18,
            23, 28, 33, 38,
            43, 48, 53, 58,
        )

        var expected_std = Tensor[dtype](3, 4, 1)
        fill[dtype, nelts](expected_std, 1.41421356)

        var B = generate_expected_tensor[12](expected_sum, 3, 4, 1)
        var C = generate_expected_tensor[12](expected_mean, 3, 4, 1)

        return SumMeanStdData(A, axis, B, C, expected_std)