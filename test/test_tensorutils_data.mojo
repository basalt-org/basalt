from tensor import Tensor, TensorShape

alias dtype = DType.float32

struct TransposeData:
    var A: Tensor[dtype]
    var expected: Tensor[dtype]
    var transpose_dims: StaticIntTuple[2]

    fn __init__(inout self, A: Tensor[dtype], expected: Tensor[dtype], transpose_dims: StaticIntTuple[2]):
        self.A = A
        self.expected = expected
        self.transpose_dims = transpose_dims

    @staticmethod
    fn generate_tensor(*shape: Int) -> Tensor[dtype]:
        var A = Tensor[dtype](shape)
        let size = A.num_elements()
        for i in range(size):
            A[i] = i + 1
        return A ^

    @staticmethod
    fn generate_expected_tensor[
        size: Int
    ](data: StaticIntTuple[size], *shape: Int) -> Tensor[dtype]:
        var A = Tensor[dtype](shape)
        for i in range(size):
            A[i] = data[i]
        return A ^

    @staticmethod
    fn generate_1_test_case() -> TransposeData:
        let A = TransposeData.generate_tensor(2, 3)
        let expected = StaticIntTuple[6](1, 4, 2, 5, 3, 6)
        let tranpose_dims = StaticIntTuple[2](0, 1)
        let B = TransposeData.generate_expected_tensor(expected, 3, 2)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_2_test_case() -> TransposeData:
        let A = TransposeData.generate_tensor(2, 3, 2)
        let expected = StaticIntTuple[12](1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12)
        let tranpose_dims = StaticIntTuple[2](0, 2)
        let B = TransposeData.generate_expected_tensor(expected, 2, 3, 2)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_3_test_case() -> TransposeData:
        let A = TransposeData.generate_tensor(2, 3, 2, 3)
        let expected = StaticIntTuple[36](1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 
        10, 11, 12, 16, 17, 18, 19, 20, 21, 25, 26, 27, 31, 32, 33, 22, 23, 24, 
        28, 29, 30, 34, 35, 36)
        let tranpose_dims = StaticIntTuple[2](1, 2)
        let B = TransposeData.generate_expected_tensor(expected, 2, 2, 3, 3)

        return TransposeData(A, B, tranpose_dims)

    @staticmethod
    fn generate_4_test_case() -> TransposeData:
        let A = TransposeData.generate_tensor(3, 2, 3, 2, 3)
        let expected = StaticIntTuple[108](1, 2, 3, 19, 20, 21, 7, 8, 9, 25, 
        26, 27, 13, 14, 15, 31, 32, 33, 4, 5, 6, 22, 23, 24, 10, 11, 12, 28, 
        29, 30, 16, 17, 18, 34, 35, 36, 37, 38, 39, 55, 56, 57, 43, 44, 45, 61, 
        62, 63, 49, 50, 51, 67, 68, 69, 40, 41, 42, 58, 59, 60, 46, 47, 48, 64, 
        65, 66, 52, 53, 54, 70, 71, 72, 73, 74, 75, 91, 92, 93, 79, 80, 81, 97, 
        98, 99, 85, 86, 87, 103, 104, 105, 76, 77, 78, 94, 95, 96, 82, 83, 84, 
        100, 101, 102, 88, 89, 90, 106, 107, 108
        )
        let tranpose_dims = StaticIntTuple[2](1, 3)
        let B = TransposeData.generate_expected_tensor(expected, 3, 2, 3, 2, 3)

        return TransposeData(A, B, tranpose_dims)
