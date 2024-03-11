from tensor import TensorShape
from time.time import now

import dainemo.nn as nn
from dainemo import Graph, Symbol, OP
from dainemo.utils.datasets import BostonHousing
from dainemo.utils.dataloader import DataLoader



fn mse(inout g: Graph, y_true: Symbol, y_pred: Symbol) -> Symbol:

    # 1/N * sum( (outputs - targets)^2 )

    var diff = g.op(OP.SUB, y_true, y_pred)
    var loss = g.op(OP.POW, diff, 2)
    var mean_loss = g.op(OP.MEAN, loss)

    return mean_loss ^


from dainemo import dtype
from dainemo.autograd.params import Param
fn par_init(shape: TensorShape) -> Param:
    var par = DynamicVector[SIMD[dtype, 1]](capacity=shape.num_elements())
    for i in range(shape.num_elements()):
        par.push_back(0.1)
    return Param(par)


fn linear_regression(batch_size: Int, n_inputs: Int, n_outputs: Int) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, n_inputs))
    var y_true = g.input(TensorShape(batch_size, n_outputs))
    

    var W = g.param(TensorShape(n_inputs, n_outputs), init="kaiming_normal")
    var b = g.param(TensorShape(n_outputs))
    var res = g.op(OP.DOT, x, W)

    var y_pred = g.op(OP.ADD, res, b)
    g.out(y_pred)

    var loss = mse(g, y_true, y_pred)
    g.loss(loss)

    return g^



fn main():
    var train_data: BostonHousing
    try:
        train_data = BostonHousing(file_path='./examples/data/housing.csv')
    except:
        print("Could not load data")
    
    # Train Parameters
    alias batch_size = 32
    alias num_epochs = 200
    alias learning_rate = 0.02

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
    var start = now()

    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        for batch in training_loader:

            # Forward pass
            var loss = model.forward(batch.data, batch.labels)
            # print(loss)

            # Backward pass
            optim.zero_grad(model.parameters)
            model.backward()
            optim.step(model.parameters)

            epoch_loss += loss[0]
            num_batches += 1

        print("Epoch: [", epoch+1, "/", num_epochs, "] \t Avg loss per epoch:", epoch_loss / num_batches)
    
    print("Training finished: ", (now() - start)/1e9, "seconds")

    # try:
    #     graph.render("operator")
    # except:
    #     print("Could not render graph")

    print("\n\nInferencing model...\n")
    for batch in training_loader:
        var output = model.inference(batch.data)

        # Print first (and only output)
        print("Predicted: ", output[0])