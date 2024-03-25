from random import rand
from testing import assert_equal
from test_tensorutils import assert_tensors_equal
from math import exp, log
from python.python import Python

import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.autograd.attributes import Attribute, AttributeVector
from basalt.utils.tensorutils import fill
from basalt.autograd.ops.ops import backward_op

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


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
    var shape = DynamicVector[Int]()
    for i in range(np_array.ndim):
        shape.push_back(np_array.shape[i].to_float64().to_int())

    var tensor = Tensor[dtype](TensorShape(shape))

    for i in range(tensor.num_elements()):
        tensor[i] = np_array.ravel()[i].to_float64().cast[dtype]()

    return tensor


# ------ Test Binary Ops ------
@value
struct torch_output_binary_op:
    var expected: Tensor[dtype]
    var grad_1: Tensor[dtype]
    var grad_2: Tensor[dtype]


fn torch_binary_op(
    op: OP, input_1: Tensor, input_2: Tensor, upper_grad: Tensor
) -> torch_output_binary_op:
    try:
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")

        var input_1 = torch.from_numpy(to_numpy(input_1)).requires_grad_(True)
        var input_2 = torch.from_numpy(to_numpy(input_2)).requires_grad_(True)

        var expected: PythonObject

        if op == OP.ADD:
            expected = input_1 + input_2
        elif op == OP.SUB:
            expected = input_1 - input_2
        elif op == OP.MUL:
            expected = input_1 * input_2
        elif op == OP.DIV:
            expected = input_1 / input_2
        elif op == OP.DOT:
            expected = torch.matmul(input_1, input_2)
        else:
            print("Error: op not supported (returning the default add op result): ", op)
            expected = input_1 + input_2

        # uppergrad & backwards
        var upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        return torch_output_binary_op(
            to_tensor(expected.detach().numpy()),
            to_tensor(input_1.grad.numpy()),
            to_tensor(input_2.grad.numpy()),
        )

    except:
        print("Error importing torch")
        var d = Tensor[dtype](1)
        return torch_output_binary_op(d, d, d)


fn test_binary_op[
    op: OP, t1_shape: TensorShape, t2_shape: TensorShape
](t1: Tensor[dtype], t2: Tensor[dtype], expected: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)
        var t2 = g.input(t2_shape)

        var res = g.op(op, t1, t2)
        g.out(res)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1, t2)[0]
    assert_tensors_equal(res, expected, "almost")


fn test_binary_op_backward[
    op: OP, t1_shape: TensorShape, t2_shape: TensorShape, ug_shape: TensorShape
](
    t1: Tensor[dtype],
    t2: Tensor[dtype],
    ug: Tensor[dtype],
    grad_1_expected: Tensor[dtype],
    grad_2_expected: Tensor[dtype],
) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, t2_shape, AttributeVector()](
        ug, t1, t2, grad_1
    )
    assert_tensors_equal(grad_1, grad_1_expected, "almost")

    var grad_2 = Tensor[dtype](t2_shape)
    backward_op[1, op, ug_shape, t1_shape, t2_shape, AttributeVector()](
        ug, t1, t2, grad_2
    )
    assert_tensors_equal(grad_2, grad_2_expected, "almost")


fn test_ADD() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias t2_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_binary_op(OP.ADD, t1, t2, ug)

    test_binary_op[OP.ADD, t1_shape, t2_shape](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.ADD, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )

    # broadcasting

    alias t1_shape_2 = TensorShape(37, 63, 107)
    alias t2_shape_2 = TensorShape(37, 63, 1)
    alias ug_shape_2 = TensorShape(37, 63, 107)

    t1 = Tensor[dtype](t1_shape_2)
    t2 = Tensor[dtype](t2_shape_2)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    ug = Tensor[dtype](ug_shape_2)
    rand(ug.data(), ug.num_elements())

    expected_and_grad = torch_binary_op(OP.ADD, t1, t2, ug)

    test_binary_op[OP.ADD, t1_shape_2, t2_shape_2](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.ADD, t1_shape_2, t2_shape_2, ug_shape_2](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )


fn test_SUB() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias t2_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_binary_op(OP.SUB, t1, t2, ug)

    test_binary_op[OP.SUB, t1_shape, t2_shape](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.SUB, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )

    # broadcasting

    alias t1_shape_2 = TensorShape(37, 63, 107)
    alias t2_shape_2 = TensorShape(37, 63, 1)
    alias ug_shape_2 = TensorShape(37, 63, 107)

    t1 = Tensor[dtype](t1_shape_2)
    t2 = Tensor[dtype](t2_shape_2)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    ug = Tensor[dtype](ug_shape_2)
    rand(ug.data(), ug.num_elements())

    expected_and_grad = torch_binary_op(OP.SUB, t1, t2, ug)

    test_binary_op[OP.SUB, t1_shape_2, t2_shape_2](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.SUB, t1_shape_2, t2_shape_2, ug_shape_2](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )


fn test_MUL() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias t2_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_binary_op(OP.MUL, t1, t2, ug)

    test_binary_op[OP.MUL, t1_shape, t2_shape](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.MUL, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )

    # broadcasting
    alias t1_shape_2 = TensorShape(37, 63, 107)
    alias t2_shape_2 = TensorShape(37, 63, 1)
    alias ug_shape_2 = TensorShape(37, 63, 107)

    t1 = Tensor[dtype](t1_shape_2)
    t2 = Tensor[dtype](t2_shape_2)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    ug = Tensor[dtype](ug_shape_2)
    rand(ug.data(), ug.num_elements())

    expected_and_grad = torch_binary_op(OP.MUL, t1, t2, ug)

    test_binary_op[OP.MUL, t1_shape_2, t2_shape_2](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.MUL, t1_shape_2, t2_shape_2, ug_shape_2](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )


fn test_DIV() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias t2_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_binary_op(OP.DIV, t1, t2, ug)

    test_binary_op[OP.DIV, t1_shape, t2_shape](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.DIV, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )

    # broadcasting
    alias t1_shape_2 = TensorShape(37, 63, 107)
    alias t2_shape_2 = TensorShape(37, 63, 1)
    alias ug_shape_2 = TensorShape(37, 63, 107)

    t1 = Tensor[dtype](t1_shape_2)
    t2 = Tensor[dtype](t2_shape_2)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    ug = Tensor[dtype](ug_shape_2)
    rand(ug.data(), ug.num_elements())

    expected_and_grad = torch_binary_op(OP.DIV, t1, t2, ug)

    test_binary_op[OP.DIV, t1_shape_2, t2_shape_2](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.DIV, t1_shape_2, t2_shape_2, ug_shape_2](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )

    alias t1_shape_3 = TensorShape(37, 63, 1)
    alias t2_shape_3 = TensorShape(37, 63, 107)
    alias ug_shape_3 = TensorShape(37, 63, 107)

    t1 = Tensor[dtype](t1_shape_3)
    t2 = Tensor[dtype](t2_shape_3)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    ug = Tensor[dtype](ug_shape_3)
    rand(ug.data(), ug.num_elements())

    expected_and_grad = torch_binary_op(OP.DIV, t1, t2, ug)

    test_binary_op[OP.DIV, t1_shape_3, t2_shape_3](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.DIV, t1_shape_3, t2_shape_3, ug_shape_3](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )


fn test_DOT() raises:
    alias t1_shape = TensorShape(107, 203)
    alias t2_shape = TensorShape(203, 139)
    alias ug_shape = TensorShape(107, 139)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_binary_op(OP.DOT, t1, t2, ug)

    test_binary_op[OP.DOT, t1_shape, t2_shape](t1, t2, expected_and_grad.expected)
    test_binary_op_backward[OP.DOT, t1_shape, t2_shape, ug_shape](
        t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    )


# ------ Test Unary Ops ------
@value
struct torch_output_unary_op:
    var expected: Tensor[dtype]
    var grad_1: Tensor[dtype]


fn torch_unary_op(op: OP, input_1: Tensor, upper_grad: Tensor) -> torch_output_unary_op:
    try:
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")

        var input_1 = torch.from_numpy(to_numpy(input_1)).requires_grad_(True)

        var expected: PythonObject

        if op == OP.EXP:
            expected = torch.exp(input_1)
        elif op == OP.LOG:
            expected = torch.log(input_1)
        elif op == OP.POW:
            expected = torch.pow(input_1, 2)
        else:
            print("Error: op not supported (returning the value input_1): ", op)
            expected = input_1

        # uppergrad & backwards
        var upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        return torch_output_unary_op(
            to_tensor(expected.detach().numpy()),
            to_tensor(input_1.grad.numpy()),
        )

    except:
        print("Error importing torch")
        var d = Tensor[dtype](1)
        return torch_output_unary_op(d, d)


fn test_unary_op[
    op: OP, t1_shape: TensorShape
](t1: Tensor[dtype], expected: Tensor[dtype]) raises:
    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(t1_shape)

        var res = g.op(op, t1)
        g.out(res)

        return g ^

    alias graph = create_graph()
    assert_equal(len(graph.nodes), 1)

    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1)[0]

    assert_tensors_equal(res, expected, "almost")


fn test_unary_op_backward[
    op: OP, t1_shape: TensorShape, ug_shape: TensorShape
](t1: Tensor[dtype], ug: Tensor[dtype], grad_1_expected: Tensor[dtype],) raises:
    var grad_1 = Tensor[dtype](t1_shape)
    backward_op[0, op, ug_shape, t1_shape, AttributeVector()](ug, t1, grad_1)
    assert_tensors_equal(grad_1, grad_1_expected, "almost")


fn test_EXP() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.EXP, t1, ug)

    test_unary_op[OP.EXP, t1_shape](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.EXP, t1_shape, ug_shape](t1, ug, expected_and_grad.grad_1)


fn test_LOG() raises:
    alias t1_shape = TensorShape(37, 63, 107)
    alias ug_shape = TensorShape(37, 63, 107)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_unary_op(OP.LOG, t1, ug)

    test_unary_op[OP.LOG, t1_shape](t1, expected_and_grad.expected)
    test_unary_op_backward[OP.LOG, t1_shape, ug_shape](t1, ug, expected_and_grad.grad_1)


fn main():
    print("Running ops (compare with torch) tests")
    try:
        test_ADD()
        test_SUB()
        test_MUL()
        test_DIV()
        test_DOT()
        test_EXP()
        test_LOG()
    except e:
        print("[ERROR] Error in ops (compare with torch)")
        print(e)
        return

    print("Finished ops (compare with torch) tests")
