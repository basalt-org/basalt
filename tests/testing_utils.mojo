from testing import assert_equal, assert_almost_equal
from collections import OptionalReg
from python.python import Python

from basalt.autograd.attributes import AttributeVector
from basalt.autograd.ops.ops import backward_op
from basalt.nn import Tensor, TensorShape
from basalt.autograd import Graph, OP
from basalt import dtype


# assert_tensors_equal\(([^,]+),\s*([^,]+),\s*"([^"]+)"\)
# assert_tensors_equal["$3"]($1, $2)
fn assert_tensors_equal[
    mode: String = "exact", msg: String = "Error"
](t1: Tensor[dtype], t2: Tensor[dtype]) raises:
    constrained[
        mode == "exact" or mode == "almost", "Mode must be either 'exact' or 'almost'"
    ]()

    assert_equal(t1.shape(), t2.shape(), "Tensor shape mismatch")

    for i in range(t1.num_elements()):
        if mode == "almost":
            assert_almost_equal(t1[i], t2[i], rtol=1e-5, atol=1e-5, msg=msg)
        else:
            assert_equal(t1[i], t2[i], msg=msg)


fn test_unary_op[
    op: OP, t1_shape: TensorShape, attrs: OptionalReg[AttributeVector] = None
](t1: Tensor[dtype], expected: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)

        if attrs:
            var res = g.op(op, t1, attributes=attrs.value())
            g.out(res)
            return g ^
        else:
            res = g.op(op, t1)
            g.out(res)
            return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1)[0]

    assert_tensors_equal["almost"](res, expected)


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

        if attrs:
            var res = g.op(op, t1, t2, attributes=attrs.value())
            g.out(res)
            return g ^
        else:
            var res = g.op(op, t1, t2)
            g.out(res)
            return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1, t2)[0]

    assert_tensors_equal["almost"](res, expected)


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

    assert_tensors_equal["almost"](res, expected)


fn test_unary_op_backward[
    op: OP,
    t1_shape: TensorShape,
    ug_shape: TensorShape,
    attrs: AttributeVector = AttributeVector(),
](t1: Tensor[dtype], ug: Tensor[dtype], grad_1_expected: Tensor[dtype],) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, attrs](ug, t1, grad_1)
    assert_tensors_equal["almost"](grad_1, grad_1_expected)


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
    assert_tensors_equal["almost"](grad_1, grad_1_expected)

    var grad_2 = Tensor[dtype](t2_shape)
    backward_op[1, op, ug_shape, t1_shape, t2_shape, attrs](ug, t1, t2, grad_2)
    assert_tensors_equal["almost"](grad_2, grad_2_expected)


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
    assert_tensors_equal["almost"](grad_1, grad_1_expected)

    var grad_2 = Tensor[dtype](t2_shape)
    backward_op[1, op, ug_shape, t1_shape, t2_shape, t3_shape, attrs](
        ug, t1, t2, t3, grad_2
    )
    assert_tensors_equal["almost"](grad_2, grad_2_expected)

    var grad_3 = Tensor[dtype](t3_shape)
    backward_op[2, op, ug_shape, t1_shape, t2_shape, t3_shape, attrs](
        ug, t1, t2, t3, grad_3
    )
    assert_tensors_equal["almost"](grad_3, grad_3_expected)


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
        shape.append(np_array.shape[i].to_float64().to_int())

    var tensor = Tensor[dtype](TensorShape(shape))

    # Calling ravel a lot of times is slow
    var np_array_temp = np_array.ravel()

    for i in range(tensor.num_elements()):
        tensor[i] = np_array_temp[i].to_float64().cast[dtype]()

    return tensor
