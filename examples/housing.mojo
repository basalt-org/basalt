from tensor import Tensor

import dainemo.nn as nn
from dainemo.autograd.node import Node
from dainemo.utils.datasets import BostonHousing
from dainemo.utils.dataloader import DataLoader, housing_data_batch, housing_label_batch

alias dtype = DType.float32



struct LinearRegression:
    var layer1: nn.Linear

    fn __init__(inout self, input_dim: Int):
        self.layer1 = nn.Linear(input_dim, 1)
        
    fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
        return self.layer1(Node[dtype](x))



fn main():    
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
    
    var model = LinearRegression(train_data.data.dim(1))
    var loss_func = nn.MSELoss()
    var optim = nn.optim.Adam(lr=0.05)

    let batch_data: Tensor[dtype]
    let batch_labels: Tensor[dtype]
    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: SIMD[dtype, 1] = 0.0
        for batch_indeces in training_loader:

            batch_data = housing_data_batch[dtype](batch_indeces.get[0, Int](), batch_indeces.get[1, Int](), training_loader.data)
            batch_labels = housing_label_batch[dtype](batch_indeces.get[0, Int](), batch_indeces.get[1, Int](), training_loader.labels)
            
            optim.zero_grad()
            let output = model.forward(batch_data) 
            print("model forward")           
            var loss = loss_func(output, batch_labels)
            print("loss done")

            loss.backward()
            print("backward done")
            optim.step()

            epoch_loss += loss.tensor[0]
            num_batches += 1
            print(num_batches)
        
        print("Epoch: [", epoch+1, "/", num_epochs, "] \t Avg loss per epoch:", epoch_loss / num_batches)