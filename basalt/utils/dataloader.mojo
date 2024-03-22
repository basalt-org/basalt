from testing import assert_equal
from math import min
from memory import memcpy

from basalt import dtype, nelts
from basalt import Tensor, TensorShape


@value
struct Batch[dtype: DType](CollectionElement):
    var data: Tensor[dtype]
    var labels: Tensor[dtype]

    fn __init__(inout self, owned data: Tensor[dtype], owned labels: Tensor[dtype]):
        self.data = data^
        self.labels = labels^

    fn __getitem__(self, index: Int) -> Tensor[dtype]:
        if index == 0:
            return self.data
        elif index == 1:
            return self.labels
        else:
            print("[ERROR] Batch.__getitem__(): Index out of bounds")
            return Tensor[dtype]()


@value
struct DataLoader:
    var data: Tensor[dtype]
    var labels: Tensor[dtype]
    var batch_size: Int
    var _current_index: Int
    var _num_batches: Int
    var _data_batch_shape: TensorShape
    var _label_batch_shape: TensorShape

    fn __init__(
            inout self, 
            data: Tensor[dtype],
            labels: Tensor[dtype],
            batch_size: Int, 
        ):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self._current_index = 0
        self._num_batches = 0
        
        # Batch shapes
        self._data_batch_shape = self.data.shape()
        self._label_batch_shape = self.labels.shape()
        self._data_batch_shape[0] = self.batch_size
        self._label_batch_shape[0] = self.batch_size

    @always_inline
    fn __len__(self) -> Int:
        '''
        Returns the number of the batches left in the dataset.
        '''
        return self._num_batches

    fn __iter__(inout self) -> Self:
        # Number of batches left, NOTE: ignore the remainder for now
        # var remainder = 1 if self.data.dim(0) % self.batch_size != 0 else 0
        var full_batches = self.data.dim(0) // self.batch_size
        self._num_batches = full_batches + 0
        self._current_index = 0
        return self

    fn __next__(inout self) -> Batch[dtype]:
        var start = self._current_index
        var end = min(self._current_index + self.batch_size, self.data.dim(0))
        
        # NOTE: ignore the remainder for now
        # self._data_shape[0] = end - start
        # self._label_shape[0] = end - start

        var data_batch = self.create_batch(self.data, self._data_batch_shape, start)
        var label_batch = self.create_batch(self.labels, self._label_batch_shape, start) 

        self._current_index += self.batch_size
        self._num_batches -= 1
        return Batch[dtype](data_batch, label_batch)

    @staticmethod
    @always_inline
    fn create_batch(tensor: Tensor[dtype], batch_shape: TensorShape, start: Int) -> Tensor[dtype]:
        var batch_tensor = Tensor[dtype](batch_shape)
        memcpy(batch_tensor.data(), tensor.data().offset(start*batch_shape.strides()[0]), batch_shape.num_elements())        
        return batch_tensor ^
