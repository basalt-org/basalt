from time.time import now

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP, dtype
from basalt.utils.datasets import MNIST
from basalt.utils.dataloader import DataLoader
from basalt.autograd.attributes import AttributeVector, Attribute


def plot_image(data: Tensor, num: Int):
    from python.python import Python, PythonObject

    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    var pyimage: PythonObject = np.empty((28, 28), np.float64)
    for m in range(28):
        for n in range(28):
            pyimage.itemset((m, n), data[num * 28 * 28 + m * 28 + n])

    plt.imshow(pyimage)
    plt.show()


fn create_CNN(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 1, 28, 28))

    var x1 = nn.Conv2d(g, x, out_channels=16, kernel_size=5, padding=2)
    var x2 = nn.ReLU(g, x1)
    var x3 = nn.MaxPool2d(g, x2, kernel_size=2)
    var x4 = nn.Conv2d(g, x3, out_channels=32, kernel_size=5, padding=2)
    var x5 = nn.ReLU(g, x4)
    var x6 = nn.MaxPool2d(g, x5, kernel_size=2)
    var x7 = g.op(
        OP.RESHAPE,
        x6,
        attributes=AttributeVector(
            Attribute(
                "shape",
                TensorShape(x6.shape[0], x6.shape[1] * x6.shape[2] * x6.shape[3]),
            )
        ),
    )
    var out = nn.Linear(g, x7, n_outputs=10)
    g.out(out)

    var y_true = g.input(TensorShape(batch_size, 10))
    var loss = nn.CrossEntropyLoss(g, out, y_true)
    # var loss = nn.MSELoss(g, out, y_true)
    g.loss(loss)

    return g ^


fn main():
    alias num_epochs = 20
    alias batch_size = 4
    alias learning_rate = 1e-3

    alias graph = create_CNN(batch_size)

    # try:
    #     graph.render("operator")
    # except:
    #     print("Could not render graph")

    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](lr=learning_rate)
    optim.allocate_rms_and_momentum(model.parameters)

    print("Loading data ...")
    var train_data: MNIST
    try:
        train_data = MNIST(file_path="./examples/data/mnist_test_small.csv")
        _ = plot_image(train_data.data, 1)
    except:
        print("Could not load data")

    var training_loader = DataLoader(
        data=train_data.data, labels=train_data.labels, batch_size=batch_size
    )

    print("Training started/")
    var start = now()

    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        var epoch_start = now()
        for batch in training_loader:
            # [ONE HOT ENCODING!]
            var labels_one_hot = Tensor[dtype](batch.labels.dim(0), 10)
            for bb in range(batch.labels.dim(0)):
                labels_one_hot[(bb * 10 + batch.labels[bb]).to_int()] = 1.0

            # Forward pass
            var loss = model.forward(batch.data, labels_one_hot)

            # Backward pass
            optim.zero_grad(model.parameters)
            model.backward()
            optim.step(model.parameters)

            epoch_loss += loss[0]
            num_batches += 1

            print(
                "Epoch [",
                epoch + 1,
                "/",
                num_epochs,
                "],\t Step [",
                num_batches,
                "/",
                train_data.data.dim(0) // batch_size,
                "],\t Loss:",
                epoch_loss / num_batches,
            )

        print("Epoch time: ", (now() - epoch_start) / 1e9, "seconds")

    print("Training finished: ", (now() - start) / 1e9, "seconds")


    model.perf_metrics.print_forward_perf_metrics("ms", True)
    model.perf_metrics.print_backward_perf_metrics("ms", True)
