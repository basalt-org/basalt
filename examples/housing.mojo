from tensor import Tensor

import dainemo.nn as nn
from dainemo.autograd.node import Node
from dainemo.utils.datasets import BostonHousing
from dainemo.utils.dataloader import DataLoader

alias dtype = DType.float32



struct LinearRegression:
    var layer1: nn.Linear

    fn __init__(inout self, input_dim: Int):
        self.layer1 = nn.Linear(input_dim, 1)
        
    fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
        return self.layer1(Node[dtype](x))



fn main():
    var train_data: BostonHousing[dtype]
    try:
        train_data = BostonHousing[dtype](file_path='./examples/data/housing.csv')
    except:
        print("Could not load data")
    
    # Train Parameters
    alias batch_size = 64
    alias num_epochs = 200
    alias learning_rate = 0.05

    # Batchwise data loader
    var training_loader = DataLoader[dtype](
                            data=train_data.data,
                            labels=train_data.labels,
                            batch_size=batch_size
                        )
    
    
    var model = LinearRegression(train_data.data.dim(1))
    var loss_func = nn.MSELoss()
    var optim = nn.optim.Adam(lr=learning_rate)


    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        for batch in training_loader:

            # Forward pass
            var output = model.forward(batch.data)
            var loss = loss_func(output, batch.labels)

            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.tensor[0]
            num_batches += 1

        print("Epoch: [", epoch+1, "/", num_epochs, "] \t Avg loss per epoch:", epoch_loss / num_batches)