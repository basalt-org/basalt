from testing import assert_equal
from math import min
from memory import memcpy

from basalt import dtype, nelts
from basalt import Tensor, TensorShape
from basalt.utils.datasets import CIFAR10

@value
struct Batch[dtype: DType](CollectionElement):
    var data: Tensor[dtype]
    var labels: Tensor[dtype]

    fn __init__(inout self, batch_data: Tensor[dtype], batch_labels: Tensor[dtype]):
        self.data = batch_data
        self.labels = batch_labels

    fn __init__(
        inout self,
        df_data: Tensor[dtype],
        df_labels: Tensor[dtype],
        start: Int,
        batch_data_shape: TensorShape,
        batch_labels_shape: TensorShape,
    ):
        # TODO: find a better way to do this
        # Links to the copies of the input tensors in model.forward()
        self.data = Tensor[dtype](batch_data_shape)
        self.labels = Tensor[dtype](batch_labels_shape)
        memcpy(
            self.data.data(),
            df_data.data().offset(start * batch_data_shape.strides()[0]),
            batch_data_shape.num_elements(),
        )
        memcpy(
            self.labels.data(),
            df_labels.data().offset(start * batch_labels_shape.strides()[0]),
            batch_labels_shape.num_elements(),
        )

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

        # Number of batches to iter, NOTE: ignore the remainder for now
        # var remainder = 1 if self.data.dim(0) % self.batch_size != 0 else 0
        self._current_index = 0
        self._num_batches = self.data.dim(0) // self.batch_size  # + remainder

        # Batch shapes
        self._data_batch_shape = self.data.shape()
        self._label_batch_shape = self.labels.shape()
        self._data_batch_shape[0] = self.batch_size
        self._label_batch_shape[0] = self.batch_size

    @always_inline
    fn __len__(self) -> Int:
        """
        Returns the number of the batches left in the dataset.
        """
        return self._num_batches

    fn __iter__(self) -> Self:
        # TODO: Starting the iterator requires to return (COPY!) the whole dataloader which containts the whole dataset
        # Does this mean that the whole dataset is copied every epoch ?!
        return self

    fn __next__(inout self) -> Batch[dtype]:
        # NOTE: ignore the remainder for now
        # var end = min(self._current_index + self.batch_size, self.data.dim(0))
        # self._data_shape[0] = end - self._current_index
        # self._label_shape[0] = end - self._current_index

        var temp_current_index = self._current_index
        self._current_index += self.batch_size
        self._num_batches -= 1

        return Batch[dtype](
            self.data,
            self.labels,
            temp_current_index,
            self._data_batch_shape,
            self._label_batch_shape,
        )

@value
struct CIFARDataLoader:
    var dataset: CIFAR10 #BaseDataset
    # The error I get is: "dynamic traits not supported yet, please use a compile time generic instead of 'BaseDataset'"
    var batch_size: Int
    var _current_index: Int
    var _num_batches: Int
    #var _data_batch_shape: TensorShape
    #var _label_batch_shape: TensorShape

    fn __init__(
        inout self,
        owned dataset: CIFAR10, #BaseDataset,
        batch_size: Int,
    ):
        self.dataset = dataset^
        self.batch_size = batch_size

        # Number of batches to iter, NOTE: ignore the remainder for now
        # var remainder = 1 if self.data.dim(0) % self.batch_size != 0 else 0
        self._current_index = 0
        self._num_batches = len(self.dataset) // self.batch_size  # + remainder


    @always_inline
    fn __len__(self) -> Int:
        """
        Returns the number of the batches left in the dataset.
        """
        return self._num_batches

    fn __iter__(self) -> Self:
        # TODO: Starting the iterator requires to return (COPY!) the whole dataloader which containts the whole dataset
        # Does this mean that the whole dataset is copied every epoch ?!
        return self

    fn __next__(inout self) raises -> Batch[dtype]:

        var init_data = self.dataset[self._current_index]

        var data = Tensor[dtype](self.batch_size, init_data[0].shape()[0], init_data[0].shape()[1], init_data[0].shape()[2])
        var labels = Tensor[dtype](self.batch_size, 1)

        var offset = 0
        var imsize = init_data[0].num_elements()
        for i in range(imsize):            
            data[i] = init_data[0][i]


        labels[0] = init_data[1]

        offset += imsize

        for i in range(1, self.batch_size):
            var next_item = self.dataset[self._current_index + i]
            for j in range(imsize):
                data[j + offset] = next_item[0][j]
            labels[i] = next_item[1]

            offset += imsize

        var temp_current_index = self._current_index
        self._current_index += self.batch_size
        self._num_batches -= 1

        return Batch[dtype](
            batch_data = data,
            batch_labels = labels
        )