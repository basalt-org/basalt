from tensor import TensorShape
from time.time import now

import dainemo.nn as nn
from dainemo import Graph, Symbol, OP
from dainemo.utils.datasets import BostonHousing
from dainemo.utils.dataloader import DataLoader



fn mse(inout g: Graph, y_true: Symbol, y_pred: Symbol) -> Symbol:

    let diff = g.op(OP.SUB, y_true, y_pred)
    let loss = g.op(OP.MUL, diff, diff)
    let mean_loss = g.op(OP.MEAN, loss, None)

    return mean_loss ^



fn linear_regression(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    let x = g.input(TensorShape(batch_size, n_inputs))
    let y_true = g.input(TensorShape(batch_size, n_outputs))

    let W = g.param(TensorShape(n_inputs, n_outputs))
    let b = g.param(TensorShape(n_outputs))
    let res = g.op(OP.DOT, x, W)

    let y_pred = g.op(OP.ADD, res, b)
    let loss = mse(g, y_true, y_pred)
    _ = g.out(loss)

    return g^



fn main():
    let train_data: BostonHousing
    try:
        train_data = BostonHousing(file_path='./examples/data/housing.csv')
    except:
        print("Could not load data")
    
    # Train Parameters
    alias batch_size = 64
    alias num_epochs = 100
    alias learning_rate = 1e-4

    # Batchwise data loader
    var training_loader = DataLoader(
                            data=train_data.data,
                            labels=train_data.labels,
                            batch_size=batch_size
                        )
    
    alias graph = linear_regression(batch_size, train_data.n_inputs, 1)
    
    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](lr=learning_rate)
    optim.allocate_rms_and_momentum(model.parameters)

    print("Training started")
    let start = now()

    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        for batch in training_loader:

            # Forward pass
            let loss = model.forward(batch.data, batch.labels)
            # print(loss)

            # Backward pass
            optim.zero_grad(model.parameters)
            model.backward()
            optim.step(model.parameters)

            epoch_loss += loss[0]
            num_batches += 1

        print("Epoch: [", epoch+1, "/", num_epochs, "] \t Avg loss per epoch:", epoch_loss / num_batches)
    
    print("Training finished: ", (now() - start)/1e9, "seconds")