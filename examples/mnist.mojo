from tensor import Tensor

import dainemo.nn as nn
from dainemo.autograd.node import Node
from dainemo.utils.datasets import MNIST
from dainemo.utils.dataloader import DataLoader, mnist_data_batch, mnist_label_batch

alias dtype = DType.float32



def plot_image[dtype: DType](borrowed data: Tensor[dtype], num: Int):
    from utils.index import Index
    from python.python import Python, PythonObject
    
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    let pyimage: PythonObject = np.empty((28, 28), np.float64)
    for m in range(28):
        for n in range(28):
            pyimage.itemset((m, n), data[Index(num, 0, m, n)])

    plt.imshow(pyimage)
    plt.show()



struct Model:
    var layer1: nn.Linear

    fn __init__(inout self):
        self.layer1 = nn.Linear(28*28, 256)
        
    fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
        return self.layer1(Node[dtype](x))



fn main():    
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
    
    var model = Model()

    let batch_data: Tensor[dtype]
    let batch_labels: Tensor[dtype]
    for epoch in range(num_epochs):
        for batch in training_loader:
                        
            # TODO: Dataloader needs FLATTEN & RESHAPE to generalize for any rank
            batch_data = mnist_data_batch[dtype](batch.start, batch.end, training_loader.data)
            batch_labels = mnist_label_batch[dtype](batch.start, batch.end, training_loader.labels)

            try:
                _ = plot_image[dtype](batch_data, 0)
            except: 
                print("Could not plot image")

            let output = model.forward(batch_data)
