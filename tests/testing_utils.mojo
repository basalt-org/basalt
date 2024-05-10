from python.python import Python
from collections import OptionalReg
from testing import assert_equal, assert_almost_equal

from basalt import dtype
from basalt.autograd import Graph, OP
from basalt.autograd.ops.ops import backward_op
from basalt.autograd.attributes import AttributeVector
from basalt.nn import Tensor, TensorShape, Model


# The below regex should be used to convert deprecated calls
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
            var res = g.op(op, t1)
            g.out(res)
            return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = Model[graph](inference_only=True)
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

    var model = Model[graph](inference_only=True)
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

    var model = Model[graph](inference_only=True)
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


fn to_numpy(tensor: Tensor) -> PythonObject:
    try:
        var np = Python.import_module("numpy")
    
        np.set_printoptions(4)

        var rank = tensor.rank()
        var dims = PythonObject([])
        for i in range(rank):
            dims.append(tensor.dim(i))
        var pyarray: PythonObject = np.empty(dims, dtype=np.float32)

        var pointer = int(pyarray.__array_interface__['data'][0].to_float64())
        var pointer_d = DTypePointer[tensor.dtype](address=pointer)
        memcpy(pointer_d, tensor.data(), tensor.num_elements())
    
        return pyarray
    except e:
        print("Error in to numpy", e)
        return PythonObject()


fn to_tensor(np_array: PythonObject) raises -> Tensor[dtype]:
    var shape = List[Int]()
    for i in range(np_array.ndim):
        shape.append(int(np_array.shape[i].to_float64()))

    var tensor = Tensor[dtype](TensorShape(shape))

    var np_array_2 = np_array.copy()
    try:
        var np = Python.import_module("numpy")
        np_array_2 = np.float32(np_array_2)
    except e:
        print("Error in to tensor", e)

    var pointer = int(np_array_2.__array_interface__['data'][0].to_float64())
    var pointer_d = DTypePointer[tensor.dtype](address=pointer)
    memcpy(tensor.data(), pointer_d, tensor.num_elements())

    _ = np_array_2

    return tensor


fn create_graph_concat(
    t1_shape: TensorShape, t2_shape: TensorShape, t3_shape: TensorShape, dim: Int
) -> Graph:
    # Testing with 3 operands
    var g = Graph()
    var t1 = g.input(t1_shape, trainable=True)
    var t2 = g.input(t2_shape, trainable=True)
    var t3 = g.input(t3_shape, trainable=True)
    var res = g.concat(t1, t2, t3, dim=dim)
    g.out(res)
    g.loss(res)
    return g ^


fn create_graph_split(t_shape: TensorShape, sections: List[Int], dim: Int) -> Graph:
    var g = Graph()
    var t = g.input(t_shape, trainable=True)
    var results = g.split(t, sections=sections, dim=dim)
    for i in range(len(sections)):
        g.out(results[i])
    g.loss(results[0])  # Any one
    return g ^
