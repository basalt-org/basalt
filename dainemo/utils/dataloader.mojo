from tensor import Tensor, TensorShape
from algorithm import parallelize, vectorize
from utils.index import Index
from random import rand
from math import min


@value
struct MyBatch[dtype: DType](CollectionElement):
    #TODO: Mojo v0.6.0: Tensor is missing the CollectionElement trait
    # Remove MyBatch and replace with DynamicVector[Tensor[dtype]]()
    var data: Tensor[dtype]
    var labels: Tensor[dtype]
    var start: Int      # TODO: Remove with FLATTEN and RESHAPE operations
    var end: Int        # TODO: Remove with FLATTEN and RESHAPE operations

    fn __init__(inout self, data: Tensor[dtype], labels: Tensor[dtype], start: Int, end: Int):
        self.data = data
        self.labels = labels
        self.start = start
        self.end = end

    fn __getitem__(self, index: Int) -> Tensor[dtype]:
        if index == 0:
            return self.data
        elif index == 1:
            return self.labels
        else:
            print("[ERROR] MyBatch.__getitem__(): Index out of bounds")
            return Tensor[dtype]()


@value
struct DataLoader[dtype: DType]:
    var data: Tensor[dtype]
    var labels: Tensor[dtype]
    var batch_size: Int
    var _current_index: Int
    var _num_batches: Int
    var _data_shape: DynamicVector[Int]
    var _label_shape: DynamicVector[Int]

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
        self._data_shape = DynamicVector[Int]()
        self._label_shape = DynamicVector[Int]()

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

    fn __next__(inout self) -> MyBatch[dtype]:
        let start = self._current_index
        let end = min(self._current_index + self.batch_size, self.data.dim(0))
        self.update_batch_shape(start, end)

        let data_batch = self.create_batch(self.data, TensorShape(self._data_shape), start)
        let label_batch = self.create_batch(self.labels, TensorShape(self._label_shape), start)
        # TODO: Store in DynamicVector[Tensor[dtype]]()
        let batch = MyBatch[dtype](data_batch, label_batch, start, end)

        self._current_index += self.batch_size
        self._num_batches -= 1
        return batch

    fn update_batch_shape(inout self, start: Int, end: Int):
        # 1. Data batch shape
        self._data_shape = DynamicVector[Int]()
        self._data_shape.push_back(end - start)
        for i in range(self.data.rank() - 1):
            self._data_shape.push_back(self.data.dim(i + 1))
        
        # 2. Label batch shape
        self._label_shape = DynamicVector[Int]()
        self._label_shape.push_back(end - start)
        for i in range(self.labels.rank() - 1):
            self._label_shape.push_back(self.labels.dim(i + 1))


    @staticmethod
    @always_inline
    fn create_batch(inout tensor: Tensor[dtype], shape: TensorShape, start: Int) -> Tensor[dtype]:
        var batch_tensor = Tensor[dtype](shape)

        # TODO: Only works for rank 2 tensors. 
        # Needs FLATTEN and RESHAPE operations to generalize to any rank.

        for n in range(batch_tensor.dim(0)):
            for i in range(batch_tensor.num_elements() // batch_tensor.dim(0)):
                batch_tensor[Index(n, i)] = tensor[Index(start + n, i)]

        return batch_tensor


## Workarounds
fn housing_data_batch[dtype: DType](start: Int, end: Int, data: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start, 13))
    for n in range(end - start):
        for i in range(13):
            batch[Index(n, i)] = data[Index(start + n, i)]
    return batch

fn housing_label_batch[dtype: DType](start: Int, end: Int, labels: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start, 1))    
    for i in range(end - start):
        batch[i] = labels[start + i]
    return batch

fn mnist_data_batch[dtype: DType](start: Int, end: Int, data: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start, 1, 28, 28))
    for i in range(end - start):
        for m in range(28):
            for n in range(28):
                batch[Index(i, 0, m, n)] = data[Index(start + i, 0, m, n)]
    return batch

fn mnist_label_batch[dtype: DType](start: Int, end: Int, labels: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start, 1))
    for i in range(end - start):
        batch[i] = labels[start + i]
    return batch