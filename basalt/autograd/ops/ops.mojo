from .basics import (
    ADD,
    SUB,
    MUL,
    DIV,
    EXP,
    LOG,
    POW,
    DOT,
    SUM,
    MEAN,
    MAX,
    FLATTEN,
    RESHAPE,
    TRANSPOSE,
    FMA,
)
from .mlops import SIGMOID, RELU, TANH, CLIP, SQUEEZE, UNSQUEEZE, CONCAT2
from .conv import CONV2D
from .pool import MAXPOOL2D

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

    alias ADD = OP(0, "ADD")
    alias SUB = OP(1, "SUB")
    alias MUL = OP(2, "MUL")
    alias DIV = OP(3, "DIV")
    alias EXP = OP(4, "EXP")
    alias LOG = OP(5, "LOG")
    alias POW = OP(6, "POW")
    alias DOT = OP(7, "DOT")
    alias SUM = OP(8, "SUM")
    alias MEAN = OP(9, "MEAN")
    alias MAX = OP(10, "MAX")
    alias FLATTEN = OP(11, "FLATTEN")
    alias RESHAPE = OP(12, "RESHAPE")
    alias SIGMOID = OP(13, "SIGMOID")
    alias RELU = OP(14, "RELU")
    alias TANH = OP(15, "TANH")
    alias CONV2D = OP(16, "CONV2D")
    alias TRANSPOSE = OP(17, "TRANSPOSE")
    alias MAXPOOL2D = OP(18, "MAXPOOL2D")
    alias FMA = OP(19, "FMA")
    alias CLIP = OP(20, "CLIP")
    alias SQUEEZE = OP(21, "SQUEEZE")
    alias UNSQUEEZE = OP(22, "UNSQUEEZE")
    alias CONCAT2 = OP(23, "CONCAT2")

    var id: UInt8
    var name: Bytes[16]
    var dynamic: Bool

    fn __init__(inout self, id: UInt8, name: String, dynamic: Bool = False):
        self.id = id
        self.name = Bytes[16](name)
        self.dynamic = dynamic

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
        return EXP.result_shape(t1_shape)
    elif op == OP.LOG:
        return LOG.result_shape(t1_shape)
    elif op == OP.SUM:
        return SUM.result_shape(t1_shape, attributes)
    elif op == OP.MEAN:
        return MEAN.result_shape(t1_shape, attributes)
    elif op == OP.MAX:
        return MAX.result_shape(t1_shape, attributes)
    elif op == OP.FLATTEN:
        return FLATTEN.result_shape(t1_shape)
    elif op == OP.RESHAPE:
        return RESHAPE.result_shape(t1_shape, attributes)
    elif op == OP.SIGMOID:
        return SIGMOID.result_shape(t1_shape)
    elif op == OP.RELU:
        return RELU.result_shape(t1_shape)
    elif op == OP.TANH:
        return TANH.result_shape(t1_shape)
    elif op == OP.TRANSPOSE:
        return TRANSPOSE.result_shape(t1_shape, attributes)
    elif op == OP.MAXPOOL2D:
        return MAXPOOL2D.result_shape(t1_shape, attributes)
    elif op == OP.CLIP:
        return CLIP.result_shape(t1_shape)
    elif op == OP.SQUEEZE:
        return SQUEEZE.result_shape(t1_shape, attributes)
    elif op == OP.UNSQUEEZE:
        return UNSQUEEZE.result_shape(t1_shape, attributes)
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
        return ADD.result_shape(t1_shape, t2_shape)
    elif op == OP.SUB:
        return SUB.result_shape(t1_shape, t2_shape)
    elif op == OP.MUL:
        return MUL.result_shape(t1_shape, t2_shape)
    elif op == OP.DIV:
        return DIV.result_shape(t1_shape, t2_shape)
    elif op == OP.POW:
        return POW.result_shape(t1_shape, t2_shape)
    elif op == OP.DOT:
        return DOT.result_shape(t1_shape, t2_shape)
    elif op == OP.CONCAT2:
        return CONCAT2.result_shape(t1_shape, t2_shape, attributes)
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
        return CONV2D.result_shape(t1_shape, t2_shape, t3_shape, attributes)
    elif op == OP.FMA:
        return FMA.result_shape(t1_shape, t2_shape, t3_shape)
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
        EXP.forward[t1_shape](res, t1)
    elif op == OP.LOG:
        LOG.forward[t1_shape](res, t1)
    elif op == OP.SUM:
        SUM.forward[t1_shape, attributes](res, t1)
    elif op == OP.MEAN:
        MEAN.forward[t1_shape, attributes](res, t1)
    elif op == OP.MAX:
        MAX.forward[t1_shape, attributes](res, t1)
    elif op == OP.FLATTEN:
        FLATTEN.forward[t1_shape](res, t1)
    elif op == OP.RESHAPE:
        RESHAPE.forward[t1_shape](res, t1)
    elif op == OP.SIGMOID:
        SIGMOID.forward[t1_shape](res, t1)
    elif op == OP.RELU:
        RELU.forward[t1_shape](res, t1)
    elif op == OP.TANH:
        TANH.forward[t1_shape](res, t1)
    elif op == OP.TRANSPOSE:
        TRANSPOSE.forward[t1_shape, attributes](res, t1)
    elif op == OP.MAXPOOL2D:
        MAXPOOL2D.forward[t1_shape, attributes](res, t1)
    elif op == OP.CLIP:
        CLIP.forward[t1_shape, attributes](res, t1)
    elif op == OP.SQUEEZE:
        SQUEEZE.forward[t1_shape, attributes](res, t1)
    elif op == OP.UNSQUEEZE:
        UNSQUEEZE.forward[t1_shape, attributes](res, t1)
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
        ADD.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.SUB:
        SUB.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.MUL:
        MUL.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.DIV:
        DIV.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.POW:
        POW.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.DOT:
        DOT.forward[t1_shape, t2_shape](res, t1, t2)
    elif op == OP.CONCAT2:
        CONCAT2.forward[t1_shape, t2_shape, attributes](res, t1, t2)
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
        CONV2D.forward[t1_shape, t2_shape, t3_shape, attributes](res, t1, t2, t3)
    elif op == OP.FMA:
        FMA.forward[t1_shape, t2_shape, t3_shape](res, t1, t2, t3)
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
        res_grad = EXP.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.LOG:
        res_grad = LOG.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SUM:
        res_grad = SUM.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MEAN:
        res_grad = MEAN.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MAX:
        res_grad = MAX.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.FLATTEN:
        res_grad = FLATTEN.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.RESHAPE:
        res_grad = RESHAPE.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.SIGMOID:
        res_grad = SIGMOID.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.RELU:
        res_grad = RELU.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.TANH:
        res_grad = TANH.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.TRANSPOSE:
        res_grad = TRANSPOSE.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.MAXPOOL2D:
        res_grad = MAXPOOL2D.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.CLIP:
        res_grad = CLIP.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.SQUEEZE:
        res_grad = SQUEEZE.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.UNSQUEEZE:
        res_grad = UNSQUEEZE.backward[ug_shape, t1_shape](ug, t1)
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
        res_grad = ADD.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.SUB:
        res_grad = SUB.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.MUL:
        res_grad = MUL.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.DIV:
        res_grad = DIV.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.POW:
        res_grad = POW.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.DOT:
        res_grad = DOT.backward[tensor_id, ug_shape, t1_shape, t2_shape](ug, t1, t2)
    elif op == OP.CONCAT2:
        res_grad = CONCAT2.backward[tensor_id, ug_shape, t1_shape, t2_shape, attributes](ug, t1, t2)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1, -1)

    fn broadcastable(op: OP) -> Bool:
        return op == OP.ADD or op == OP.SUB or op == OP.MUL or op == OP.DIV

    @parameter
    if tensor_id == 0:
        alias res_grad_shape = t1_shape if not broadcastable(op) else broadcast_shapes(
            t1_shape, t2_shape
        )
        # grad_shape = t1_shape
        # NOTE: Assumption res_grad.shape() == res_grad_shape
        # if res_grad.shape() != res_grad_shape:
        #     print("[ERROR] tensor_id: 0, Assumption not holding. res_grad_shape != res_grad.shape(), for binary operator.")
        accumulate_grad[t1_shape, res_grad_shape](grad, res_grad)

    elif tensor_id == 1:
        alias res_grad_shape = t2_shape if not broadcastable(op) else broadcast_shapes(
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
        res_grad = CONV2D.backward[
            tensor_id, ug_shape, t1_shape, t2_shape, t3_shape, attributes
        ](ug, t1, t2, t3)
    elif op == OP.FMA:
        res_grad = FMA.backward[tensor_id, ug_shape, t1_shape, t2_shape, t3_shape](
            ug, t1, t2, t3
        )
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
