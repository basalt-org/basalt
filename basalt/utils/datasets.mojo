from algorithm import vectorize

from basalt import dtype
from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import elwise_op, tmean, tstd


@always_inline
fn div[dtype: DType, simd_width: Int](a: SIMD[dtype, simd_width], b: Scalar[dtype]) -> SIMD[dtype, simd_width]:
    return a / b


struct BostonHousing:
    alias n_inputs = 13

    var data: Tensor[dtype]
    var labels: Tensor[dtype]

    fn __init__(inout self, file_path: String) raises:
        var s = read_file(file_path)
        # Skip the first and last lines
        # This does assume your last line in the file has a newline at the end
        var list_of_lines = s.split("\n")[1:-1]

        # Length is number of lines
        var N = len(list_of_lines)

        self.data = Tensor[dtype](N, self.n_inputs)  # All columns except the last one
        self.labels = Tensor[dtype](N, 1)  # Only the last column (MEDV)

        var line: List[String] = List[String]()

        # Load data in Tensor
        for item in range(N):
            line = list_of_lines[item].split(",")
            self.labels[item] = cast_string[dtype](line[-1])

            for n in range(self.n_inputs):
                self.data[item * self.n_inputs + n] = cast_string[dtype](line[n])

        # Normalize data
        # TODO: redo when tensorutils tmean2 and tstd2 are implemented
        alias nelts = simdwidthof[dtype]()
        var col = Tensor[dtype](N)
        for j in range(self.n_inputs):
            for k in range(N):
                col[k] = self.data[k * self.n_inputs + j]
            for i in range(N):
                self.data[i * self.n_inputs + j] = (self.data[i * self.n_inputs + j] - tmean(col)) / tstd(col)


struct MNIST:
    var data: Tensor[dtype]
    var labels: Tensor[dtype]

    fn __init__(inout self, file_path: String) raises:
        var s = read_file(file_path)
        # Skip the first and last lines
        # This does assume your last line in the file has a newline at the end
        var list_of_lines = s.split("\n")[1:-1]

        # Length is number of lines
        var N = len(list_of_lines)
        self.data = Tensor[dtype](N, 1, 28, 28)
        self.labels = Tensor[dtype](N)

        var line: List[String] = List[String]()

        # Load data in Tensor
        for item in range(N):
            line = list_of_lines[item].split(",")
            self.labels[item] = atol(line[0])
            for i in range(self.data.shape()[2]):
                for j in range(self.data.shape()[3]):
                    self.data[item * 28 * 28 + i * 28 + j] = atol(line[i * 28 + j + 1])

        # Normalize data
        alias nelts = simdwidthof[dtype]()

        @parameter
        fn vecdiv[nelts: Int](idx: Int):
            self.data.store[nelts](idx, div(self.data.load[nelts](idx), 255.0))

        vectorize[vecdiv, nelts](self.data.num_elements())


fn read_file(file_path: String) raises -> String:
    var s: String
    with open(file_path, "r") as f:
        s = f.read()
    return s


fn find_first(s: String, delimiter: String) -> Int:
    for i in range(len(s)):
        if s[i] == delimiter:
            return i
    return -1


fn cast_string[dtype: DType](s: String) raises -> Scalar[dtype]:
    """
    Cast a string with decimal to a SIMD vector of dtype.
    """

    var idx = find_first(s, delimiter=".")
    var x: Scalar[dtype] = -1

    if idx == -1:
        # No decimal point
        x = atol(s)
        return x
    else:
        var c_int: Scalar[dtype]
        var c_frac: Scalar[dtype]
        c_int = atol(s[:idx])
        c_frac = atol(s[idx + 1 :])
        x = c_int + c_frac / (10 ** len(s[idx + 1 :]))
        return x
