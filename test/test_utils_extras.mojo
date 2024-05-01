from python.python import Python
from collections.optional import OptionalReg
from testing import assert_equal
from test_tensorutils import assert_tensors_equal

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.autograd.ops.ops import backward_op
from basalt.autograd.attributes import AttributeVector, Attribute


alias dtype = DType.float32


def to_numpy(tensor: Tensor) -> PythonObject:
    var np = Python.import_module("numpy")
    np.set_printoptions(4)

    rank = tensor.rank()
    var pyarray: PythonObject = np.array([0])
    if rank == 1:
        pyarray = np.empty((tensor.dim(0)))
    elif rank == 2:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1)))
    elif rank == 3:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2)))
    elif rank == 4:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3)))
    else:
        print("Error: rank not supported: ", rank)

    for i in range(tensor.num_elements()):
        pyarray.itemset((i), tensor[i])

    return pyarray


fn to_tensor(np_array: PythonObject) raises -> Tensor[dtype]:
    var shape = List[Int]()
    for i in range(np_array.ndim):
        shape.append(int(np_array.shape[i].to_float64()))

    var tensor = Tensor[dtype](TensorShape(shape))

    # Calling ravel a lot of times is slow
    var np_array_temp = np_array.ravel()

    for i in range(tensor.num_elements()):
        tensor[i] = np_array_temp[i].to_float64().cast[dtype]()

    return tensor


# ------- Test forward ops -------
fn test_unary_op[
    op: OP, t1_shape: TensorShape, attrs: OptionalReg[AttributeVector] = None
](t1: Tensor[dtype], expected: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)

        var res: Symbol
        if attrs:
            res = g.op(op, t1, attributes=attrs.value())
        else:
            res = g.op(op, t1)
        g.out(res)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1)[0]

    assert_tensors_equal(res, expected, "almost")


fn test_binary_op[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attrs: OptionalReg[AttributeVector] = None,
](t1: Tensor[dtype], t2: Tensor[dtype], expected: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)
        var t2 = g.input(t2_shape)

        var res: Symbol
        if attrs:
            res = g.op(op, t1, t2, attributes=attrs.value())
        else:
            res = g.op(op, t1, t2)
        g.out(res)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1, t2)[0]

    assert_tensors_equal(res, expected, "almost")


fn test_ternary_op[
    op: OP, t1_shape: TensorShape, t2_shape: TensorShape, t3_shape: TensorShape
](
    t1: Tensor[dtype], t2: Tensor[dtype], t3: Tensor[dtype], expected: Tensor[dtype]
) raises:
    @parameter
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)
        var t2 = g.input(t2_shape)
        var t3 = g.input(t3_shape)

        var res = g.op(op, t1, t2, t3)
        g.out(res)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1, t2, t3)[0]

    assert_tensors_equal(res, expected, "almost")


# ------- Test backward ops -------
fn test_unary_op_backward[
    op: OP,
    t1_shape: TensorShape,
    ug_shape: TensorShape,
    attrs: AttributeVector = AttributeVector(),
](t1: Tensor[dtype], ug: Tensor[dtype], grad_1_expected: Tensor[dtype],) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, attrs](ug, t1, grad_1)
    assert_tensors_equal(grad_1, grad_1_expected, "almost")


fn test_binary_op_backward[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    ug_shape: TensorShape,
    attrs: AttributeVector = AttributeVector(),
](
    t1: Tensor[dtype],
    t2: Tensor[dtype],
    ug: Tensor[dtype],
    grad_1_expected: Tensor[dtype],
    grad_2_expected: Tensor[dtype],
) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, t2_shape, attrs](ug, t1, t2, grad_1)
    assert_tensors_equal(grad_1, grad_1_expected, "almost")

    var grad_2 = Tensor[dtype](t2_shape)
    backward_op[1, op, ug_shape, t1_shape, t2_shape, attrs](ug, t1, t2, grad_2)
    assert_tensors_equal(grad_2, grad_2_expected, "almost")


fn test_ternary_op_backward[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    ug_shape: TensorShape,
    attrs: AttributeVector = AttributeVector(),
](
    t1: Tensor[dtype],
    t2: Tensor[dtype],
    t3: Tensor[dtype],
    ug: Tensor[dtype],
    grad_1_expected: Tensor[dtype],
    grad_2_expected: Tensor[dtype],
    grad_3_expected: Tensor[dtype],
) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, t2_shape, t3_shape, attrs](
        ug, t1, t2, t3, grad_1
    )
    assert_tensors_equal(grad_1, grad_1_expected, "almost")

    var grad_2 = Tensor[dtype](t2_shape)
    backward_op[1, op, ug_shape, t1_shape, t2_shape, t3_shape, attrs](
        ug, t1, t2, t3, grad_2
    )
    assert_tensors_equal(grad_2, grad_2_expected, "almost")

    var grad_3 = Tensor[dtype](t3_shape)
    backward_op[2, op, ug_shape, t1_shape, t2_shape, t3_shape, attrs](
        ug, t1, t2, t3, grad_3
    )
    assert_tensors_equal(grad_3, grad_3_expected, "almost")
