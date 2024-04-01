from .basics import (
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Log,
    Pow,
    Dot,
    Sum,
    Mean,
    Max,
    Flatten,
    Reshape,
    Transpose,
)
from .mlops import Sigmoid, Relu, Tanh
from .conv import Conv_2D
from .pool import MAXPOOL2D

from basalt import Tensor, TensorShape
from basalt.utils.bytes import bytes
from basalt.utils.tensorutils import broadcast_shapes, accumulate_grad
from ..attributes import AttributeVector


# Define operators as named parameter expression
@value
@register_passable("trivial")
struct OP(Stringable):
    """
    Compile time Operators list.
    """

    alias ADD = OP(0, "ADD", num_operands=2)
    alias SUB = OP(1, "SUB", num_operands=2)
    alias MUL = OP(2, "MUL", num_operands=2)
    alias DIV = OP(3, "DIV", num_operands=2)
    alias EXP = OP(4, "EXP", num_operands=1)
    alias LOG = OP(5, "LOG", num_operands=1)
    alias POW = OP(6, "POW", num_operands=2)
    alias DOT = OP(7, "DOT", num_operands=2)
    alias SUM = OP(8, "SUM", num_operands=1)
    alias MEAN = OP(9, "MEAN", num_operands=1)
    alias MAX = OP(10, "MAX", num_operands=1)
    alias FLATTEN = OP(11, "FLATTEN", num_operands=1)
    alias RESHAPE = OP(12, "RESHAPE", num_operands=1)
    alias SIGMOID = OP(13, "SIGMOID", num_operands=1)
    alias RELU = OP(14, "RELU", num_operands=1)
    alias TANH = OP(15, "TANH", num_operands=1)
    alias CONV2D = OP(16, "CONV2D", num_operands=3)
    alias TRANSPOSE = OP(17, "TRANSPOSE", num_operands=1)
    alias MAXPOOL2D = OP(18, "MAXPOOL2D", num_operands=1)

    var id: UInt8
    var name: bytes[16]
    var num_operands: UInt8

    fn __init__(inout self, id: UInt8, name: String, num_operands: UInt8):
        self.id = id
        self.name = bytes[16](name)
        self.num_operands = num_operands

    fn __eq__(self, other: OP) -> Bool:
        return self.id == other.id

    fn __str__(self) -> String:
        return str(self.name)


fn static_result_shape(
    op: OP, t1_shape: TensorShape, attributes: AttributeVector
) -> TensorShape:
    """
    Static result shape for unary operators.
    """
    if op == OP.EXP:
        return Exp.result_shape(t1_shape)
    elif op == OP.LOG:
        return Log.result_shape(t1_shape)
    elif op == OP.SUM:
        return Sum.result_shape(t1_shape, attributes)
    elif op == OP.MEAN:
        return Mean.result_shape(t1_shape, attributes)
    elif op == OP.MAX:
        return Max.result_shape(t1_shape, attributes)
    elif op == OP.FLATTEN:
        return Flatten.result_shape(t1_shape)
    elif op == OP.RESHAPE:
        return Reshape.result_shape(t1_shape, attributes)
    elif op == OP.SIGMOID:
        return Sigmoid.result_shape(t1_shape)
    elif op == OP.RELU:
        return Relu.result_shape(t1_shape)
    elif op == OP.TANH:
        return Tanh.result_shape(t1_shape)
    elif op == OP.TRANSPOSE:
        return Transpose.result_shape(t1_shape, attributes)
    elif op == OP.MAXPOOL2D:
        return MAXPOOL2D.result_shape(t1_shape, attributes)
    else:
        print("[ERROR] Operator not found.")
        return TensorShape(-1)


fn static_result_shape(
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
) -> TensorShape:
    """
    Static result shape for binary operators.
    """
    if op == OP.ADD:
        return Add.result_shape(t1_shape, t2_shape)
    elif op == OP.SUB:
        return Sub.result_shape(t1_shape, t2_shape)
    elif op == OP.MUL:
        return Mul.result_shape(t1_shape, t2_shape)
    elif op == OP.DIV:
        return Div.result_shape(t1_shape, t2_shape)
    elif op == OP.POW:
        return Pow.result_shape(t1_shape, t2_shape)
    elif op == OP.DOT:
        return Dot.result_shape(t1_shape, t2_shape)
    else:
        # We can't print at compile time (at least for now it crashes at comp time with an error)
        print("[ERROR] Operator not found.")
        return TensorShape(-1, -1)


fn static_result_shape(
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
) -> TensorShape:
    """
    Static result shape for ternary operators.
    """

    if op == OP.CONV2D:
        return Conv_2D.result_shape(t1_shape, t2_shape, t3_shape, attributes)
    else:
        print("[ERROR] Operator not found.")
        return TensorShape(-1, -1)


fn forward_op[
    op: OP, t1_shape: TensorShape, attributes: AttributeVector
](inout res: Tensor[dtype], t1: Tensor[dtype]):
    """
    Forward pass for unary operators.
    """

    @parameter
    if op == OP.EXP:
        Exp.forward[t1_shape](res, t1)
    elif op == OP.LOG:
        Log.forward[t1_shape](res, t1)
    elif op == OP.SUM:
        Sum.forward[t1_shape, attributes](res, t1)
    elif op == OP.MEAN:
        Mean.forward[t1_shape, attributes](res, t1)
    elif op == OP.MAX:
        Max.forward[t1_shape, attributes](res, t1)
    elif op == OP.FLATTEN:
        Flatten.forward[t1_shape](res, t1)
    elif op == OP.RESHAPE:
        Reshape.forward[t1_shape](res, t1)
    elif op == OP.SIGMOID:
        Sigmoid.forward[t1_shape](res, t1)
    elif op == OP.RELU:
        Relu.forward[t1_shape](res, t1)
    elif op == OP.TANH:
        Tanh.forward[t1_shape](res, t1)
    elif op == OP.TRANSPOSE:
        Transpose.forward[t1_shape, attributes](res, t1)
    elif op == OP.MAXPOOL2D:
        MAXPOOL2D.forward[t1_shape, attributes](res, t1)
    else:
        print("[ERROR] Operator not found.")


fn forward_op[
    op: OP, t1_shape: TensorShape, t2_shape: TensorShape, attributes: AttributeVector
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    """
    Forward pass for binary operators.
    """

    @parameter
    if op == OP.ADD:
        Add.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.SUB:
        Sub.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.MUL:
        Mul.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.DIV:
        Div.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.POW:
        Pow.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.DOT:
        Dot.forward[t1_shape, t2_shape](res, t1, t2)
    else:
        print("[ERROR] Operator not found.")


fn forward_op[
    op: OP,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype], t3: Tensor[dtype]):
    """
    Forward pass for ternary operators.
    """

    @parameter
    if op == OP.CONV2D:
        Conv_2D.forward[t1_shape, t2_shape, t3_shape, attributes](res, t1, t2, t3)
    else:
        print("[ERROR] Operator not found.")


fn backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    attributes: AttributeVector,
](ug: Tensor[dtype], t1: Tensor[dtype], inout grad: Tensor[dtype]):
    """
    Backward pass for unary operators.
    """
    var res_grad: Tensor[dtype]

    @parameter
    if op == OP.EXP:
        res_grad = Exp.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.LOG:
        res_grad = Log.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SUM:
        res_grad = Sum.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MEAN:
        res_grad = Mean.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MAX:
        res_grad = Max.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.FLATTEN:
        res_grad = Flatten.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.RESHAPE:
        res_grad = Reshape.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SIGMOID:
        res_grad = Sigmoid.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.RELU:
        res_grad = Relu.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.TANH:
        res_grad = Tanh.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.TRANSPOSE:
        res_grad = Transpose.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MAXPOOL2D:
        res_grad = MAXPOOL2D.backward[ug_shape, t1_shape, attributes](ug, t1)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1)

    alias res_grad_shape = t1_shape
    # grad_shape = t1_shape
    # NOTE: Assumption res_grad.shape() == res_grad_shape
    # if res_grad.shape() != res_grad_shape:
    #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for unary operator.")
    accumulate_grad[t1_shape, res_grad_shape](grad, res_grad)


fn backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    attributes: AttributeVector,
](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype], inout grad: Tensor[dtype]):
    """
    Backward pass for binary operators.
    """
    var res_grad: Tensor[dtype]

    @parameter
    if op == OP.ADD:
        res_grad = Add.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.SUB:
        res_grad = Sub.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.MUL:
        res_grad = Mul.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.DIV:
        res_grad = Div.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.POW:
        res_grad = Pow.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.DOT:
        res_grad = Dot.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1, -1)

    @parameter
    if tensor_id == 0:
        alias res_grad_shape = t1_shape if op == OP.DOT else broadcast_shapes(
            t1_shape, t2_shape
        )
        # grad_shape = t1_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for binary operator.")
        accumulate_grad[t1_shape, res_grad_shape](grad, res_grad)

    elif tensor_id == 1:
        alias res_grad_shape = t2_shape if op == OP.DOT else broadcast_shapes(
            t1_shape, t2_shape
        )
        # grad_shape = t2_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 1, Assumption not holding. res_grad_shape != res_grad.shape(), for binary operator.")
        accumulate_grad[t2_shape, res_grad_shape](grad, res_grad)


fn backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    t2_shape: TensorShape,
    t3_shape: TensorShape,
    attributes: AttributeVector,
](
    ug: Tensor[dtype],
    t1: Tensor[dtype],
    t2: Tensor[dtype],
    t3: Tensor[dtype],
    inout grad: Tensor[dtype],
):
    """
    Backward pass for ternary operators.
    """
    var res_grad: Tensor[dtype]

    @parameter
    if op == OP.CONV2D:
        res_grad = Conv_2D.backward[
            tensor_id, ug_shape, t1_shape, t2_shape, t3_shape, attributes
        ](ug, t1, t2, t3)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1, -1)

    @parameter
    if tensor_id == 0:
        alias res_grad_shape = t1_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for ternary operator.")
        accumulate_grad[t1_shape, res_grad_shape](grad, res_grad)
    elif tensor_id == 1:
        alias res_grad_shape = t2_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for ternary operator.")
        accumulate_grad[t2_shape, res_grad_shape](grad, res_grad)
    elif tensor_id == 2:
        alias res_grad_shape = t3_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for ternary operator.")
        accumulate_grad[t3_shape, res_grad_shape](grad, res_grad)
