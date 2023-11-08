from tensor import Tensor, TensorShape
from utils.index import Index
from random import rand
from math import min


struct DataLoader[dtype: DType]:
    var data: Tensor[dtype]
    var labels: Tensor[dtype]
    var batch_size: Int
    var _current_index: Int
    var _num_batches: Int

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

    @always_inline
    fn __copyinit__(inout self, other: DataLoader[dtype]):
        self.data = other.data
        self.labels = other.labels
        self.batch_size = other.batch_size
        self._current_index = other._current_index
        self._num_batches = other._num_batches

    @always_inline
    fn __len__(self) -> Int:
        '''
        Returns the number of the batches left in the dataset.
        '''
        return self._num_batches

    fn __iter__(inout self) -> DataLoader[dtype]:
        self._current_index = 0
        let full_batches = self.data.dim(0) // self.batch_size
        let remainder = 1 if self.data.dim(0) % self.batch_size != 0 else 0
        self._num_batches = full_batches + remainder
        return self

    fn __next__(inout self) -> (Int, Int):
        let start = self._current_index
        let end = min(self._current_index + self.batch_size, self.data.dim(0))

        # TODO: (current: mojo v0.4.0) create data batch & label batch here and return as (Tensor[dtype], Tensor[dtype]])
        # 1. Not supported yet: Can't get tensor out of tuple (Traits: https://github.com/modularml/mojo/issues/516)
        # 2. Not support yet: get 2 elements when looping over the iter metod (eg for batch, label_batch in dataloader)
        # 3. Can't dynamically creating a TensorShape 
        #    e.g TensorShape(end - start, 1, 28, 28) for mnsit and 
        #    e.g TensorShape(end - start, 13) for boston housing
        
        # Workaround: return start and end and create batch per fixed example

        self._current_index += self.batch_size
        self._num_batches -= 1
        return start, end

        