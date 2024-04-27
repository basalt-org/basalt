from random import rand
from python import Python
from math.limit import max_finite
from testing import assert_almost_equal
from test_conv import to_numpy, to_tensor
from test_tensorutils import assert_tensors_equal

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP, dtype
from basalt.utils.rand_utils import MersenneTwister


fn create_simple_nn(
    batch_size: Int,
    linear1_weights: List[SIMD[dtype, 1]],
    linear1_bias: List[SIMD[dtype, 1]],
    linear2_weights: List[SIMD[dtype, 1]],
    linear2_bias: List[SIMD[dtype, 1]],
    linear3_weights: List[SIMD[dtype, 1]],
    linear3_bias: List[SIMD[dtype, 1]],
) -> Graph:
    var g = Graph()

    var x = g.input(TensorShape(batch_size, 1))
    var y_true = g.input(TensorShape(batch_size, 1))

    # Linear 1: nn.Linear(g, x, n_outputs=32)
    var l1_w = g.param(TensorShape(1, 32), init=linear1_weights)
    var l1_b = g.param(TensorShape(32), init=linear1_bias)
    var res_1 = g.op(OP.DOT, x, l1_w)
    var x1 = g.op(OP.ADD, res_1, l1_b)

    # ReLU 1
    var x2 = nn.ReLU(g, x1)

    # Linear 2: nn.Linear(g, x2, n_outputs=32)
    var l2_w = g.param(TensorShape(32, 32), init=linear2_weights)
    var l2_b = g.param(TensorShape(32), init=linear2_bias)
    var res_2 = g.op(OP.DOT, x2, l2_w)
    var x3 = g.op(OP.ADD, res_2, l2_b)

    # ReLU 2
    var x4 = nn.ReLU(g, x3)

    # Linear 3: nn.Linear(g, x4, n_outputs=1)
    var l3_w = g.param(TensorShape(32, 1), init=linear3_weights)
    var l3_b = g.param(TensorShape(1), init=linear3_bias)
    var res_3 = g.op(OP.DOT, x4, l3_w)
    var y_pred = g.op(OP.ADD, res_3, l3_b)
    g.out(y_pred)

    var loss = nn.MSELoss(g, y_pred, y_true)
    g.loss(loss)

    return g ^


fn run_mojo[
    batch_size: Int,
    linear1_weights: List[SIMD[dtype, 1]],
    linear1_bias: List[SIMD[dtype, 1]],
    linear2_weights: List[SIMD[dtype, 1]],
    linear2_bias: List[SIMD[dtype, 1]],
    linear3_weights: List[SIMD[dtype, 1]],
    linear3_bias: List[SIMD[dtype, 1]],
](
    epochs: Int,
    learning_rate: Float64,
    inputs: Tensor[dtype],
    labels: Tensor[dtype],
) -> List[SIMD[dtype, 1]]:
    alias graph = create_simple_nn(
        batch_size,
        linear1_weights,
        linear1_bias,
        linear2_weights,
        linear2_bias,
        linear3_weights,
        linear3_bias,
    )

    var model = nn.Model[graph]()
    var optim = nn.optim.Adam[graph](Reference(model.parameters), lr=learning_rate)

    var losses = List[SIMD[dtype, 1]]()

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
    owned linear1_weights: Tensor,
    owned linear1_bias: Tensor,
    owned linear2_weights: Tensor,
    owned linear2_bias: Tensor,
    owned linear3_weights: Tensor,
    owned linear3_bias: Tensor,
) -> List[SIMD[dtype, 1]]:
    var out: List[SIMD[dtype, 1]] = List[SIMD[dtype, 1]]()

    try:
        var torch = Python.import_module("torch")
        var F = Python.import_module("torch.nn.functional")
        var np = Python.import_module("numpy")
        Python.add_to_path("./test")
        var torch_models = Python.import_module("test_models_torch")

        var inputs = torch.from_numpy(to_numpy(inputs)).requires_grad_(True)
        var labels = torch.from_numpy(to_numpy(labels)).requires_grad_(True)

        var linear1_weights = torch.from_numpy(
            to_numpy(linear1_weights)
        ).requires_grad_(True)
        var linear1_bias = torch.from_numpy(to_numpy(linear1_bias)).requires_grad_(True)
        var linear2_weights = torch.from_numpy(
            to_numpy(linear2_weights)
        ).requires_grad_(True)
        var linear2_bias = torch.from_numpy(to_numpy(linear2_bias)).requires_grad_(True)
        var linear3_weights = torch.from_numpy(
            to_numpy(linear3_weights)
        ).requires_grad_(True)
        var linear3_bias = torch.from_numpy(to_numpy(linear3_bias)).requires_grad_(True)

        var regression = torch_models.SimpleNN(
            linear1_weights,
            linear1_bias,
            linear2_weights,
            linear2_bias,
            linear3_weights,
            linear3_bias,
        )

        var loss_func = torch_models.MSELoss()
        var optimizer = torch.optim.Adam(regression.parameters(), learning_rate)

        for i in range(epochs):
            var output = regression.forward(inputs)
            var loss = loss_func(output, labels)

            _ = optimizer.zero_grad()
            _ = loss.backward()
            _ = optimizer.step()

            out.append(to_tensor(loss)[0].cast[dtype]())

        return out

    except e:
        print("Error importing torch")
        print(e)
        return out


fn create_weights(num_elements: Int, zero: Bool) -> List[SIMD[dtype, 1]]:
    var prng = MersenneTwister(123456)
    var weights = List[SIMD[dtype, 1]](capacity=num_elements)
    for i in range(num_elements):
        if zero:
            weights.append(SIMD[dtype, 1](0.0))
        else:
            var rand_float = prng.next().cast[dtype]() / max_finite[DType.int32]().cast[
                dtype
            ]()
            weights.append(SIMD[dtype, 1](rand_float / 10))
    return weights ^


fn dv_to_tensor(dv: List[SIMD[dtype, 1]], shape: TensorShape) -> Tensor[dtype]:
    var t = Tensor[dtype](shape)
    if t.num_elements() != len(dv):
        print("[WARNING] tensor and dv not the shame shape")
    for i in range(t.num_elements()):
        t[i] = dv[i]
    return t ^


fn main():
    alias learning_rate = 1e-3
    alias epochs = 100
    alias batch_size = 64
    alias n_outputs = 10

    var x_data = Tensor[dtype](batch_size, 1)
    rand[dtype](x_data.data(), x_data.num_elements())
    var y_data = Tensor[dtype](batch_size, 1)
    for j in range(batch_size):
        x_data[j] = x_data[j] * 2 - 1
        y_data[j] = math.sin(x_data[j])

    alias l1_w_shape = TensorShape(1, 32)
    alias l1_b_shape = TensorShape(32)
    alias l2_w_shape = TensorShape(32, 32)
    alias l2_b_shape = TensorShape(32)
    alias l3_w_shape = TensorShape(32, 1)
    alias l3_b_shape = TensorShape(1)

    alias linear1_weights = create_weights(l1_w_shape.num_elements(), zero=False)
    alias linear1_bias = create_weights(l1_b_shape.num_elements(), zero=False)
    alias linear2_weights = create_weights(l2_w_shape.num_elements(), zero=False)
    alias linear2_bias = create_weights(l2_b_shape.num_elements(), zero=False)
    alias linear3_weights = create_weights(l3_w_shape.num_elements(), zero=False)
    alias linear3_bias = create_weights(l3_b_shape.num_elements(), zero=False)

    var losses_mojo = run_mojo[
        batch_size,
        linear1_weights,
        linear1_bias,
        linear2_weights,
        linear2_bias,
        linear3_weights,
        linear3_bias,
    ](epochs, learning_rate, x_data, y_data)

    var losses_torch = run_torch(
        epochs,
        learning_rate,
        x_data,
        y_data,
        dv_to_tensor(linear1_weights, l1_w_shape),
        dv_to_tensor(linear1_bias, l1_b_shape),
        dv_to_tensor(linear2_weights, l2_w_shape),
        dv_to_tensor(linear2_bias, l2_b_shape),
        dv_to_tensor(linear3_weights, l3_w_shape),
        dv_to_tensor(linear3_bias, l3_b_shape),
    )

    var success = True
    for i in range(epochs):
        var loss_mojo = losses_mojo[i]
        var loss_torch = losses_torch[i]
        # print("loss_mojo: ", loss_mojo, " loss_torch: ", loss_torch)
        try:
            assert_almost_equal(loss_mojo, loss_torch, rtol=1e-4)
        except e:
            print("Losses not equal")
            print(e)
            success = False
            break

    if success:
        print("SUCCES: All losses in Sin estimate model are equal.")
