from random import rand
from python import Python
from testing import assert_almost_equal
from tests import assert_tensors_equal, to_numpy, to_tensor

from basalt import dtype
from basalt.nn import (
    Tensor,
    TensorShape,
    Model,
    ReLU,
    MaxPool2d,
    CrossEntropyLoss,
    optim,
)
from basalt.autograd import Graph, OP
from basalt.autograd.attributes import AttributeVector, Attribute


fn create_CNN(
    batch_size: Int,
    conv1_weights: List[Scalar[dtype]],
    conv1_bias: List[Scalar[dtype]],
    conv2_weights: List[Scalar[dtype]],
    conv2_bias: List[Scalar[dtype]],
    linear1_weights: List[Scalar[dtype]],
    linear1_bias: List[Scalar[dtype]],
) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 1, 28, 28))

    # conv1
    # var x1 = nn.Conv2d(g, x, out_channels=16, kernel_size=5, padding=2)
    var c1_w = g.param(TensorShape(16, x.shape[1], 5, 5), init=conv1_weights)
    var c1_b = g.param(TensorShape(16), init=conv1_bias)
    var x1 = g.op(
        OP.CONV2D,
        x,
        c1_w,
        c1_b,
        attributes=AttributeVector(
            Attribute("padding", StaticIntTuple[2](2, 2)),
            Attribute("stride", StaticIntTuple[2](1, 1)),
            Attribute("dilation", StaticIntTuple[2](1, 1)),
        ),
    )

    var x2 = ReLU(g, x1)
    var x3 = MaxPool2d(g, x2, kernel_size=2)

    # conv2
    # var x4 = nn.Conv2d(g, x3, out_channels=32, kernel_size=5, padding=2)
    var c2_w = g.param(TensorShape(32, x3.shape[1], 5, 5), init=conv2_weights)
    var c2_b = g.param(TensorShape(32), init=conv2_bias)
    var x4 = g.op(
        OP.CONV2D,
        x3,
        c2_w,
        c2_b,
        attributes=AttributeVector(
            Attribute("padding", StaticIntTuple[2](2, 2)),
            Attribute("stride", StaticIntTuple[2](1, 1)),
            Attribute("dilation", StaticIntTuple[2](1, 1)),
        ),
    )

    var x5 = ReLU(g, x4)
    var x6 = MaxPool2d(g, x5, kernel_size=2)
    var x6_shape = x6.shape
    var x7 = g.op(
        OP.RESHAPE,
        x6,
        attributes=AttributeVector(
            Attribute(
                "shape",
                TensorShape(x6_shape[0], x6_shape[1] * x6_shape[2] * x6_shape[3]),
            )
        ),
    )

    # linear1
    # var out = nn.Linear(g, x7, n_outputs=10)
    var l1_w = g.param(TensorShape(x7.shape[1], 10), init=linear1_weights)
    var l1_b = g.param(TensorShape(10), init=linear1_bias)
    var res = g.op(OP.DOT, x7, l1_w)
    var out = g.op(OP.ADD, res, l1_b)
    g.out(out)

    var y_true = g.input(TensorShape(batch_size, 10))
    var loss = CrossEntropyLoss(g, out, y_true)
    # var loss = nn.MSELoss(g, out, y_true)
    g.loss(loss)

    return g ^


fn run_mojo[
    batch_size: Int,
    conv1_weights: List[Scalar[dtype]],
    conv1_bias: List[Scalar[dtype]],
    conv2_weights: List[Scalar[dtype]],
    conv2_bias: List[Scalar[dtype]],
    linear1_weights: List[Scalar[dtype]],
    linear1_bias: List[Scalar[dtype]],
](
    epochs: Int,
    learning_rate: Float64,
    inputs: Tensor[dtype],
    labels: Tensor[dtype],
) -> List[Scalar[dtype]]:
    alias graph = create_CNN(
        batch_size,
        conv1_weights,
        conv1_bias,
        conv2_weights,
        conv2_bias,
        linear1_weights,
        linear1_bias,
    )

    var model = Model[graph]()
    var optim = optim.Adam[graph](Reference(model.parameters), lr=learning_rate)

    var losses = List[Scalar[dtype]]()

    for i in range(epochs):
        var loss = model.forward(inputs, labels)

        # Backward pass
        optim.zero_grad()
        model.backward()
        optim.step()

        losses.append(loss[0])

    return losses


fn run_torch(
    epochs: Int,
    learning_rate: Float64,
    inputs: Tensor,
    labels: Tensor,
    owned conv1_weights: Tensor,
    owned conv1_bias: Tensor,
    owned conv2_weights: Tensor,
    owned conv2_bias: Tensor,
    owned linear1_weights: Tensor,
    owned linear1_bias: Tensor,
) -> List[Scalar[dtype]]:
    var out: List[Scalar[dtype]] = List[Scalar[dtype]]()

    try:
        var torch = Python.import_module("torch")
        var F = Python.import_module("torch.nn.functional")
        var np = Python.import_module("numpy")
        Python.add_to_path("./test")
        var torch_models = Python.import_module("test_models_torch")

        var inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)
        var labels = torch.from_numpy(to_numpy(labels)).requires_grad_(True)

        var conv1_weights = torch.from_numpy(to_numpy(conv1_weights)).requires_grad_(
            True
        )
        var conv1_bias = torch.from_numpy(to_numpy(conv1_bias)).requires_grad_(True)
        var conv2_weights = torch.from_numpy(to_numpy(conv2_weights)).requires_grad_(
            True
        )
        var conv2_bias = torch.from_numpy(to_numpy(conv2_bias)).requires_grad_(True)
        var linear1_weights = torch.from_numpy(
            to_numpy(linear1_weights)
        ).requires_grad_(True)
        var linear1_bias = torch.from_numpy(to_numpy(linear1_bias)).requires_grad_(True)

        var cnn = torch_models.CNN(
            conv1_weights,
            conv1_bias,
            conv2_weights,
            conv2_bias,
            linear1_weights,
            linear1_bias,
        )

        var loss_func = torch_models.CrossEntropyLoss2()
        # var loss_func = torch.nn.CrossEntropyLoss()
        var optimizer = torch.optim.Adam(cnn.parameters(), learning_rate)

        for i in range(epochs):
            var output = cnn.forward(inputs)
            var loss = loss_func(output, labels)

            _ = optimizer.zero_grad()
            _ = loss.backward()
            _ = optimizer.step()

            out.append(to_tensor(loss)[0])

        return out

    except e:
        print("Error importing torch")
        print(e)
        return out


fn create_weights(num_elements: Int, zero: Bool) -> List[Scalar[dtype]]:
    var weights = List[Scalar[dtype]](capacity=num_elements)
    for i in range(num_elements):
        if zero:
            weights.append(Scalar[dtype](0.0))
        else:
            weights.append(Scalar[dtype](0.02))
    return weights ^


fn dv_to_tensor(dv: List[Scalar[dtype]], shape: TensorShape) -> Tensor[dtype]:
    var t = Tensor[dtype](shape)
    if t.num_elements() != len(dv):
        print("[WARNING] tensor and dv not the shame shape")
    for i in range(t.num_elements()):
        t[i] = dv[i]
    return t ^


fn main():
    alias learning_rate = 1e-3
    alias epochs = 100
    alias batch_size = 4

    var inputs = Tensor[dtype](batch_size, 1, 28, 28)
    rand[dtype](inputs.data(), inputs.num_elements())
    var labels = Tensor[dtype](batch_size, 10)  # one-hot encoded (probabilities)
    for i in range(4):
        labels[i * 10 + i] = 1.0

    alias cv1_w_shape = TensorShape(16, 1, 5, 5)
    alias conv1_weights = create_weights(cv1_w_shape.num_elements(), zero=False)
    alias cv1_b_shape = TensorShape(16)
    alias conv1_bias = create_weights(16, zero=True)

    alias cv2_w_shape = TensorShape(32, 16, 5, 5)
    alias conv2_weights = create_weights(cv2_w_shape.num_elements(), zero=False)
    alias cv2_b_shape = TensorShape(32)
    alias conv2_bias = create_weights(32, zero=True)

    alias l1_w_shape = TensorShape(32 * 7 * 7, 10)
    alias linear1_weights = create_weights(l1_w_shape.num_elements(), zero=False)
    alias l1_b_shape = TensorShape(10)
    alias linear1_bias = create_weights(10, zero=True)

    var losses_mojo = run_mojo[
        batch_size,
        conv1_weights,
        conv1_bias,
        conv2_weights,
        conv2_bias,
        linear1_weights,
        linear1_bias,
    ](
        epochs,
        learning_rate,
        inputs,
        labels,
    )

    var losses_torch = run_torch(
        epochs,
        learning_rate,
        inputs,
        labels,
        dv_to_tensor(conv1_weights, cv1_w_shape),
        dv_to_tensor(conv1_bias, cv1_b_shape),
        dv_to_tensor(conv2_weights, cv2_w_shape),
        dv_to_tensor(conv2_bias, cv2_b_shape),
        dv_to_tensor(linear1_weights, l1_w_shape),
        dv_to_tensor(linear1_bias, l1_b_shape),
    )

    for i in range(epochs):
        print("loss_mojo: ", losses_mojo[i], " loss_torch: ", losses_torch[i])

    for i in range(epochs):
        var loss_mojo = losses_mojo[i]
        var loss_torch = losses_torch[i]
        print("loss_mojo: ", loss_mojo, " loss_torch: ", loss_torch)
        try:
            assert_almost_equal(loss_mojo, loss_torch, rtol=1e-5)
        except e:
            print("Losses not equal")
            print(e)
            break
