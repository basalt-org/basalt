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
from .pool import Maxpool_2D

from basalt import Tensor, TensorShape
from basalt.utils.bytes import Bytes
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
    var name: Bytes[16]
    var num_operands: UInt8

    fn __init__(inout self, id: UInt8, name: String, num_operands: UInt8):
        self.id = id
        self.name = Bytes[16](name)
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
        return Maxpool_2D.result_shape(t1_shape, attributes)
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
    Operation: OP, FirstShape: TensorShape, Attributes: AttributeVector
](inout res: Tensor[dtype], t1: Tensor[dtype]):
    """
    Forward pass for unary operators.
    """

    @parameter
    if Operation == OP.EXP:
        Exp.forward[FirstShape](res, t1)
    elif Operation == OP.LOG:
        Log.forward[FirstShape](res, t1)
    elif Operation == OP.SUM:
        Sum.forward[FirstShape, Attributes](res, t1)
    elif Operation == OP.MEAN:
        Mean.forward[FirstShape, Attributes](res, t1)
    elif Operation == OP.MAX:
        Max.forward[FirstShape, Attributes](res, t1)
    elif Operation == OP.FLATTEN:
        Flatten.forward[FirstShape](res, t1)
    elif Operation == OP.RESHAPE:
        Reshape.forward[FirstShape](res, t1)
    elif Operation == OP.SIGMOID:
        Sigmoid.forward[FirstShape](res, t1)
    elif Operation == OP.RELU:
        Relu.forward[FirstShape](res, t1)
    elif Operation == OP.TANH:
        Tanh.forward[FirstShape](res, t1)
    elif Operation == OP.TRANSPOSE:
        Transpose.forward[FirstShape, Attributes](res, t1)
    elif Operation == OP.MAXPOOL2D:
        Maxpool_2D.forward[FirstShape, Attributes](res, t1)
    else:
        print("[ERROR] Operator not found.")


fn forward_op[
    Operation: OP,
    FirstShape: TensorShape,
    SecondShape: TensorShape,
    Attributes: AttributeVector,
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    """
    Forward pass for binary operators.
    """

    @parameter
    if Operation == OP.ADD:
        Add.forward[FirstShape, SecondShape](res, t1, t2)
    elif Operation == OP.SUB:
        Sub.forward[FirstShape, SecondShape](res, t1, t2)
    elif Operation == OP.MUL:
        Mul.forward[FirstShape, SecondShape](res, t1, t2)
    elif Operation == OP.DIV:
        Div.forward[FirstShape, SecondShape](res, t1, t2)
    elif Operation == OP.POW:
        Pow.forward[FirstShape, SecondShape](res, t1, t2)
    elif Operation == OP.DOT:
        Dot.forward[FirstShape, SecondShape](res, t1, t2)
    else:
        print("[ERROR] Operator not found.")


fn forward_op[
    Operation: OP,
    FirstShape: TensorShape,
    SecondShape: TensorShape,
    ThirdShape: TensorShape,
    Attributes: AttributeVector,
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype], t3: Tensor[dtype]):
    """
    Forward pass for ternary operators.
    """

    @parameter
    if Operation == OP.CONV2D:
        Conv_2D.forward[FirstShape, SecondShape, ThirdShape, Attributes](
            res, t1, t2, t3
        )
    else:
        print("[ERROR] Operator not found.")


fn backward_op[
    TensorID: Int,
    Operation: OP,
    UGShape: TensorShape,
    FirstShape: TensorShape,
    Attributes: AttributeVector,
](ug: Tensor[dtype], t1: Tensor[dtype], inout grad: Tensor[dtype]):
    """
    Backward pass for unary operators.
    """
    var res_grad: Tensor[dtype]

    @parameter
    if Operation == OP.EXP:
        res_grad = Exp.backward[UGShape, FirstShape](ug, t1)
    elif Operation == OP.LOG:
        res_grad = Log.backward[UGShape, FirstShape](ug, t1)
    elif Operation == OP.SUM:
        res_grad = Sum.backward[UGShape, FirstShape, Attributes](ug, t1)
    elif Operation == OP.MEAN:
        res_grad = Mean.backward[UGShape, FirstShape, Attributes](ug, t1)
    elif Operation == OP.MAX:
        res_grad = Max.backward[UGShape, FirstShape, Attributes](ug, t1)
    elif Operation == OP.FLATTEN:
        res_grad = Flatten.backward[UGShape, FirstShape](ug, t1)
    elif Operation == OP.RESHAPE:
        res_grad = Reshape.backward[UGShape, FirstShape](ug, t1)
    elif Operation == OP.SIGMOID:
        res_grad = Sigmoid.backward[UGShape, FirstShape](ug, t1)
    elif Operation == OP.RELU:
        res_grad = Relu.backward[UGShape, FirstShape](ug, t1)
    elif Operation == OP.TANH:
        res_grad = Tanh.backward[UGShape, FirstShape](ug, t1)
    elif Operation == OP.TRANSPOSE:
        res_grad = Transpose.backward[UGShape, FirstShape, Attributes](ug, t1)
    elif Operation == OP.MAXPOOL2D:
        res_grad = Maxpool_2D.backward[UGShape, FirstShape, Attributes](ug, t1)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1)

    alias res_grad_shape = FirstShape
    # grad_shape = t1_shape
    # NOTE: Assumption res_grad.shape() == res_grad_shape
    # if res_grad.shape() != res_grad_shape:
    #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for unary operator.")
    accumulate_grad[FirstShape, res_grad_shape](grad, res_grad)


fn backward_op[
    TensorID: Int,
    Operation: OP,
    UGShape: TensorShape,
    FirstShape: TensorShape,
    SecondShape: TensorShape,
    Attributes: AttributeVector,
](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype], inout grad: Tensor[dtype]):
    """
    Backward pass for binary operators.
    """
    var res_grad: Tensor[dtype]

    @parameter
    if Operation == OP.ADD:
        res_grad = Add.backward[TensorID, UGShape, FirstShape, SecondShape](ug, t1, t2)
    elif Operation == OP.SUB:
        res_grad = Sub.backward[TensorID, UGShape, FirstShape, SecondShape](ug, t1, t2)
    elif Operation == OP.MUL:
        res_grad = Mul.backward[TensorID, UGShape, FirstShape, SecondShape](ug, t1, t2)
    elif Operation == OP.DIV:
        res_grad = Div.backward[TensorID, UGShape, FirstShape, SecondShape](ug, t1, t2)
    elif Operation == OP.POW:
        res_grad = Pow.backward[TensorID, UGShape, FirstShape, SecondShape](ug, t1, t2)
    elif Operation == OP.DOT:
        res_grad = Dot.backward[TensorID, UGShape, FirstShape, SecondShape](ug, t1, t2)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1, -1)

    @parameter
    if TensorID == 0:
        alias res_grad_shape = FirstShape if Operation == OP.DOT else broadcast_shapes(
            FirstShape, SecondShape
        )
        # grad_shape = t1_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for binary operator.")
        accumulate_grad[FirstShape, res_grad_shape](grad, res_grad)

    elif TensorID == 1:
        alias res_grad_shape = SecondShape if Operation == OP.DOT else broadcast_shapes(
            FirstShape, SecondShape
        )
        # grad_shape = t2_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 1, Assumption not holding. res_grad_shape != res_grad.shape(), for binary operator.")
        accumulate_grad[SecondShape, res_grad_shape](grad, res_grad)


fn backward_op[
    TensorID: Int,
    Operation: OP,
    UGShape: TensorShape,
    FirstShape: TensorShape,
    SecondShape: TensorShape,
    ThirdShape: TensorShape,
    Attributes: AttributeVector,
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
    if Operation == OP.CONV2D:
        res_grad = Conv_2D.backward[
            TensorID, UGShape, FirstShape, SecondShape, ThirdShape, Attributes
        ](ug, t1, t2, t3)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1, -1)

    @parameter
    if TensorID == 0:
        alias res_grad_shape = FirstShape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for ternary operator.")
        accumulate_grad[FirstShape, res_grad_shape](grad, res_grad)
    elif TensorID == 1:
        alias res_grad_shape = SecondShape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for ternary operator.")
        accumulate_grad[SecondShape, res_grad_shape](grad, res_grad)
    elif TensorID == 2:
        alias res_grad_shape = ThirdShape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for ternary operator.")
        accumulate_grad[ThirdShape, res_grad_shape](grad, res_grad)
