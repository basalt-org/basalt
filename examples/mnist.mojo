from dainemo.utils.datasets import MNIST
from dainemo.utils.dataloader import DataLoader
from dainemo.utils.tensorutils import tprint

import dainemo.nn as nn

from tensor import Tensor, TensorShape
from utils.index import Index




def plot_image[dtype: DType](borrowed data: Tensor[dtype], num: Int):
    from python.python import Python, PythonObject
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    let pyimage: PythonObject = np.empty((28, 28), np.float64)
    for m in range(28):
        for n in range(28):
            pyimage.itemset((m, n), data[Index(num, 0, m, n)])

    plt.imshow(pyimage)
    plt.show()



struct Model[dtype: DType]:
    var layer1: nn.Linear[dtype]

    fn __init__(inout self):
        self.layer1 = nn.Linear[dtype](28*28, 256)

    fn forward(inout self, x: Tensor[dtype]) -> Tensor[dtype]:
        return self.layer1(x)



fn main():
    alias dtype = DType.float32
    
    let train_data: MNIST[dtype]
    try:
        train_data = MNIST[dtype](file_path='./examples/data/mnist_test_small.csv')
        _ = plot_image[dtype](train_data.data, 1)
    except:
        print("Could not load data")


    alias num_epochs = 1
    alias batch_size = 4
    var training_loader = DataLoader[dtype](
                            data=train_data.data,
                            labels=train_data.labels,
                            batch_size=batch_size
                        )
    

    var model = Model[dtype]()

    let output: Tensor[dtype]
    let batch_start: Int
    let batch_end: Int
    let batch_data: Tensor[dtype]
    let batch_labels: Tensor[dtype]
    for epoch in range(num_epochs):
        for batch_indeces in training_loader:
            
            batch_start = batch_indeces.get[0, Int]()
            batch_end = batch_indeces.get[1, Int]()
            
            batch_data = create_data_batch[dtype](batch_start, batch_end, training_loader.data)
            batch_labels = create_label_batch[dtype](batch_start, batch_end, training_loader.labels)
            
            # tprint[dtype](batch_data)
            # tprint[dtype](batch_labels)

            # try:
            #     _ = plot_image[dtype](batch_data, 0)
            # except: 
            #     print("Could not plot image")

            output = model.forward(batch_data)

            # tprint[dtype](output)
        #     break
        # break



#TODO: See DataLoader
fn create_data_batch[dtype: DType](start: Int, end: Int, data: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start, 1, 28, 28))
    for i in range(end - start):
        for m in range(28):
            for n in range(28):
                batch[Index(i, 0, m, n)] = data[Index(start + i, 0, m, n)]
    return batch

fn create_label_batch[dtype: DType](start: Int, end: Int, labels: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start))    
    for i in range(end - start):
        batch[i] = labels[start + i]
    return batch

