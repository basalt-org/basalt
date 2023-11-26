from dainemo.utils.datasets import BostonHousing
from dainemo.utils.dataloader import DataLoader, housing_data_batch, housing_label_batch

import dainemo.nn as nn
from dainemo.autograd.graph import Graph
from dainemo.autograd.node import Node

from tensor import Tensor, TensorShape
from utils.index import Index



struct LinearRegression[dtype: DType]:
    var graph: Graph[dtype]
    var layer1: nn.Linear[dtype]

    fn __init__(inout self, input_dim: Int):
        self.graph = Graph[dtype]()
        self.layer1 = nn.Linear[dtype](self.graph, input_dim, 1)
        
    fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
        return self.layer1(self.graph, Node[dtype](x))



fn main():
    alias dtype = DType.float32
    
    let train_data: BostonHousing[dtype]
    try:
        train_data = BostonHousing[dtype](file_path='./examples/data/housing.csv')
    except:
        print("Could not load data")
    
    alias num_epochs = 200
    alias batch_size = 64
    var training_loader = DataLoader[dtype](
                            data=train_data.data,
                            labels=train_data.labels,
                            batch_size=batch_size
                        )
    
    var model = LinearRegression[dtype](train_data.data.dim(1))
    var loss_func = nn.MSELoss[dtype]()
    var optim = nn.optim.Adam[dtype](lr=0.05)

    let batch_data: Tensor[dtype]
    let batch_labels: Tensor[dtype]
    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: SIMD[dtype, 1] = 0.0
        for batch_indeces in training_loader:

            batch_data = housing_data_batch[dtype](batch_indeces.get[0, Int](), batch_indeces.get[1, Int](), training_loader.data)
            batch_labels = housing_label_batch[dtype](batch_indeces.get[0, Int](), batch_indeces.get[1, Int](), training_loader.labels)
            
            optim.zero_grad(model.graph)
            let output = model.forward(batch_data)            
            var loss = loss_func(model.graph, output, batch_labels)

            loss.backward(model.graph)
            optim.step(model.graph)

            epoch_loss += loss.tensor[0]
            num_batches += 1
        
        print("Epoch: [", epoch+1, "/", num_epochs, "] \t Avg loss per epoch:", epoch_loss / num_batches)