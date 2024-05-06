from algorithm import vectorize
from math import div
import os
from pathlib import Path

from basalt import dtype
from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import elwise_op, tmean, tstd

import mimage

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



trait BaseDataset(Sized, Copyable, Movable):
    fn __getitem__(self, idx: Int) raises -> Tuple[Tensor[dtype], Int]: ...


from tensor import TensorShape as _TensorShape




struct CIFAR10(BaseDataset):
    var labels: List[Int]
    var file_paths: List[String]
    

    fn __init__(inout self, image_folder: String, label_file: String) raises:
        self.labels = List[Int]()
        self.file_paths = List[String]()

        var label_dict = Dict[String, Int]()

        with open(label_file, 'r') as f: 
            var label_list = f.read().split("\n")
            for i in range(len(label_list)):
                label_dict[label_list[i]] = i
        
        var files = os.listdir(image_folder)
        var file_dir = Path(image_folder)


        for i in range(len(files)):
            self.file_paths.append(file_dir / files[i])
            self.labels.append(label_dict[files[i].split("_")[1].split(".")[0]])

    fn __copyinit__(inout self, other: CIFAR10):
        self.labels = other.labels
        self.file_paths = other.file_paths

    # Do I need the ^ here?
    fn __moveinit__(inout self, owned other: CIFAR10):
        self.labels = other.labels^
        self.file_paths = other.file_paths^
    
    fn __len__(self) -> Int:
        return len(self.file_paths)

    fn __getitem__(self, idx: Int) raises -> Tuple[Tensor[dtype], Int]:
        var img = mimage.imread(self.file_paths[idx])

        # This does not do the correct thing!
        var imb_b = img.reshape(_TensorShape(3, 32, 32))
        img = imb_b

        # Create Basalt tensor
        var data = Tensor[dtype](img.shape()[0], img.shape()[1], img.shape()[2])

        # Differenttypes, so different SIMDwidth. 
        # How to deal? 
        """
        # Normalize data and copy from Mojo tensor to basalt tensor
        alias nelts = simdwidthof[dtype]()

        @parameter
        fn vecdiv[nelts: Int](vec_index: Int):
            data.store[nelts](vec_index, div(img.load[nelts](vec_index).cast[dtype](), 255.0))

        vectorize[vecdiv, nelts](img.num_elements())
        """
        for i in range(img.num_elements()):
            data.store(i, img.load(i).cast[dtype]()/255.0)

        return Tuple(data, self.labels[idx]) 



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
