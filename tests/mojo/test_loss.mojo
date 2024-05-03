from testing import assert_equal, assert_almost_equal

from basalt import dtype, nelts
from basalt.autograd import Graph, Symbol, OP
from basalt.nn import Model, Tensor, TensorShape, MSELoss, CrossEntropyLoss
from basalt.utils.tensorutils import fill


fn test_MSE_perfect() raises:
    alias y_pred_shape = TensorShape(2, 10)  # batch of 2, 10 classes
    alias y_true_shape = TensorShape(2, 10)

    fn create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = MSELoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 3)

    var y_pred = Tensor[dtype](y_pred_shape)
    var y_true = Tensor[dtype](y_true_shape)

    fill(y_pred, 1)
    fill(y_true, 1)

    var model = Model[graph](inference_only=True)

    var loss = model.inference(y_pred, y_true)[0]

    assert_equal(loss.dim(0), 1)  # MSE summed over all elements
    assert_equal(loss[0], 0)  # loss is 0


fn test_MSE_imperfect() raises:
    alias y_pred_shape = TensorShape(1, 10)  # batch of 1, 10 classes
    alias y_true_shape = TensorShape(1, 10)

    fn create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = MSELoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 3)

    var y_pred = Tensor[dtype](y_pred_shape)
    var y_true = Tensor[dtype](y_true_shape)

    fill(y_pred, 1)

    for i in range(10):
        y_true[i] = i

    var model = Model[graph](inference_only=True)

    var loss = model.inference(y_pred, y_true)[0]

    var expected_loss: Scalar[dtype] = 0.0

    for i in range(10):
        expected_loss += (y_pred[i] - y_true[i]) ** 2

    expected_loss = expected_loss / y_true_shape[1]

    assert_almost_equal(loss[0], expected_loss)


fn test_CrossEntropy_perfect() raises:
    alias y_pred_shape = TensorShape(2, 3)  # batch of 2, 3 classes
    alias y_true_shape = TensorShape(2, 3)

    fn create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = CrossEntropyLoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 9)

    var y_pred = Tensor[dtype](y_pred_shape)
    var y_true = Tensor[dtype](y_true_shape)

    y_pred[0 * y_pred.dim(1) + 0] = 0.1
    y_pred[0 * y_pred.dim(1) + 1] = 0.2
    y_pred[0 * y_pred.dim(1) + 2] = 0.7
    y_true[0 * y_true.dim(1) + 0] = 0
    y_true[0 * y_true.dim(1) + 1] = 0
    y_true[0 * y_true.dim(1) + 2] = 1

    y_pred[1 * y_pred.dim(1) + 0] = 0.7
    y_pred[1 * y_pred.dim(1) + 1] = 0.2
    y_pred[1 * y_pred.dim(1) + 2] = 0.1
    y_true[1 * y_true.dim(1) + 0] = 1
    y_true[1 * y_true.dim(1) + 1] = 0
    y_true[1 * y_true.dim(1) + 2] = 0

    var model = Model[graph](inference_only=True)

    var loss = model.inference(y_pred, y_true)[0]

    assert_equal(loss.shape(), TensorShape(1))
    assert_almost_equal(loss[0], 0.76794958)


fn test_CrossEntropy_imperfect() raises:
    alias y_pred_shape = TensorShape(2, 3)  # batch of 2, 3 classes
    alias y_true_shape = TensorShape(2, 3)

    fn create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = CrossEntropyLoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    alias graph = create_graph()

    var y_pred = Tensor[dtype](y_pred_shape)
    var y_true = Tensor[dtype](y_true_shape)

    y_pred[0 * y_pred.dim(1) + 0] = 0.1
    y_pred[0 * y_pred.dim(1) + 1] = 0.2
    y_pred[0 * y_pred.dim(1) + 2] = 0.7
    y_true[0 * y_true.dim(1) + 0] = 0
    y_true[0 * y_true.dim(1) + 1] = 1
    y_true[0 * y_true.dim(1) + 2] = 0

    y_pred[1 * y_pred.dim(1) + 0] = 0.7
    y_pred[1 * y_pred.dim(1) + 1] = 0.2
    y_pred[1 * y_pred.dim(1) + 2] = 0.1
    y_true[1 * y_true.dim(1) + 0] = 0
    y_true[1 * y_true.dim(1) + 1] = 0
    y_true[1 * y_true.dim(1) + 2] = 1

    var model = Model[graph](inference_only=True)

    var loss = model.inference(y_pred, y_true)[0]

    assert_equal(loss.shape(), TensorShape(1))
    assert_almost_equal(loss[0], 1.31794953)


fn main():
    try:
        test_MSE_perfect()
        test_MSE_imperfect()
        test_CrossEntropy_perfect()
        test_CrossEntropy_imperfect()
    except e:
        print("[ERROR] Error in loss")
        print(e)
