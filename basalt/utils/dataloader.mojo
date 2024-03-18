from tensor import TensorShape
from testing import assert_equal
from algorithm import parallelize, vectorize
from utils.index import Index
from math import min

from basalt import dtype


@value
struct MyBatch[dtype: DType](CollectionElement):
    #TODO: Mojo v0.6.0: Tensor is missing the CollectionElement trait
    # Remove MyBatch and replace with DynamicVector[Tensor[dtype]]()
    var data: Tensor[dtype]
    var labels: Tensor[dtype]

    fn __init__(inout self, data: Tensor[dtype], labels: Tensor[dtype]):
        self.data = data
        self.labels = labels

    fn __getitem__(self, index: Int) -> Tensor[dtype]:
        if index == 0:
            return self.data
        elif index == 1:
            return self.labels
        else:
            print("[ERROR] MyBatch.__getitem__(): Index out of bounds")
            return Tensor[dtype]()


@value
struct DataLoader:
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

        # Store original input shapes
        self._data_shape = DynamicVector[Int]()
        self._label_shape = DynamicVector[Int]()
        for i in range(self.data.rank()):
            self._data_shape.push_back(self.data.dim(i))       
        for i in range(self.labels.rank()):
            self._label_shape.push_back(self.labels.dim(i))

        # Handle data as a (total x flattened data element) tensor
        var total = data.shape()[0]
        try:
            assert_equal(labels.shape()[0], total)
            self.data.ireshape(TensorShape(total, data.num_elements() // total))
            self.labels.ireshape(TensorShape(total, labels.num_elements() // total))
        except:
            print("[ERROR] Dataloader initialization reshape failed")

    @always_inline
    fn __len__(self) -> Int:
        '''
        Returns the number of the batches left in the dataset.
        '''
        return self._num_batches

    fn __iter__(inout self) -> Self:
        self._current_index = 0
        var full_batches = self.data.dim(0) // self.batch_size
        # var remainder = 1 if self.data.dim(0) % self.batch_size != 0 else 0
        var remainder = 0
        self._num_batches = full_batches + remainder
        return self

    fn __next__(inout self) -> MyBatch[dtype]:
        var start = self._current_index
        var end = min(self._current_index + self.batch_size, self.data.dim(0))
        
        self._data_shape[0] = end - start
        self._label_shape[0] = end - start

        var data_batch = self.create_batch(self.data, TensorShape(self._data_shape), start)
        var label_batch = self.create_batch(self.labels, TensorShape(self._label_shape), start) 
        
        self._current_index += self.batch_size
        self._num_batches -= 1
        return MyBatch[dtype](data_batch, label_batch)


    @staticmethod
    @always_inline
    fn create_batch(tensor: Tensor[dtype], shape: TensorShape, start: Int) -> Tensor[dtype]:
        var batch_tensor = Tensor[dtype](shape[0], shape.num_elements() // shape[0])
        alias nelts: Int = simdwidthof[dtype]()

        @parameter
        fn calc_row(n: Int):
            
            @parameter
            fn calc_batch[nelts: Int](i: Int):
                batch_tensor.simd_store[nelts](
                    Index(n, i),
                    tensor.simd_load[nelts](Index(start + n, i))
                )

            vectorize[calc_batch, nelts](shape.num_elements() // shape[0])

        parallelize[calc_row](shape[0], shape[0])

        try:
            batch_tensor.ireshape(shape)
        except:
            print("[ERROR] Batch reshape failed")

        return batch_tensor
