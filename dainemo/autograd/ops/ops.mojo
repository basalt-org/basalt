from tensor import TensorShape

from .basics import ADD, SUB, MUL, DIV, EXP, LOG, POW, DOT, MEAN, FLATTEN, SUM
from dainemo.utils.uuid import bytes
from dainemo.utils.tensorutils import unbroadcast_add, broadcast_shapes
from ..node import Attribute, AttributeVector


# Define operators as named parameter expression
@value
@register_passable
struct OP:
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
    alias FLATTEN = OP(10, "FLATTEN")

    var id: UInt8
    var name: bytes[8]

    fn __init__(inout self, id: UInt8, name: String):
        self.id = id
        self.name = bytes[8](name)

    fn __eq__(self, other: OP) -> Bool:
        return self.id == other.id


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
        return MEAN.result_shape(t1_shape)
    elif op == OP.FLATTEN:
        return FLATTEN.result_shape(t1_shape)
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
    else:
        # We can't print at compile time (at least for now it crashes at comp time with an error)
        print("[ERROR] Operator not found.")
        return TensorShape(-1, -1)


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
    else:
        print("[ERROR] Operator not found.")


# Maybe have a special reduce operator?
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
    elif op == OP.MEAN:
        MEAN.forward[t1_shape](res, t1)
    elif op == OP.SUM:
        SUM.forward[t1_shape, attributes](res, t1)
    elif op == OP.FLATTEN:
        FLATTEN.forward[t1_shape](res, t1)
    else:
        print("[ERROR] Operator not found.")


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
    var res_grad: Tensor[dtype]  # Resulting gradient of the operation

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
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1, -1)

    @parameter
    if tensor_id == 0:

        @parameter
        fn get_res_shape() -> TensorShape:
            # We should have special broadcast shapes function for matmul and dot operations or maybe the ops class could have a function to return the shape of the backward result for id 0 and 1
            @parameter
            if op == OP.DOT:
                return t1_shape
            else:
                return broadcast_shapes(t1_shape, t2_shape)

        alias res_grad_shape = get_res_shape()

        # grad_shape = t1_shape
        unbroadcast_add[t1_shape, res_grad_shape](grad, res_grad)
    elif tensor_id == 1:

        @parameter
        fn get_res_shape_2() -> TensorShape:
            # We should have special broadcast shapes function for matmul and dot operations
            @parameter
            if op == OP.DOT:
                return t2_shape
            else:
                return broadcast_shapes(t1_shape, t2_shape)

        alias res_grad_shape = get_res_shape_2()

        # grad_shape = t2_shape
        # ug_shape != res_grad.shape(), ug_shape is equal to the result of the forward function not the backward function
        unbroadcast_add[t2_shape, res_grad_shape](grad, res_grad)


fn backward_op[
    tensor_id: Int,
    op: OP,
    ug_shape: TensorShape,
    t1_shape: TensorShape,
    attributes: AttributeVector,
](ug: Tensor[dtype], t1: Tensor[dtype], inout grad: Tensor[dtype]):
    """
    Backward pass for binary operators.
    """
    var res_grad: Tensor[dtype]  # Resulting gradient of the operation

    @parameter
    if op == OP.EXP:
        res_grad = EXP.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.LOG:
        res_grad = LOG.backward[ug_shape, t1_shape](ug, t1)
    elif op == OP.MEAN:
        res_grad = MEAN.backward[ug_shape, t1_shape](ug, t1)
    # elif op == OP.SUM:
    #     res_grad = SUM.backward[ug_shape, t1_shape, attributes](ug, t1)
    elif op == OP.FLATTEN:
        res_grad = FLATTEN.backward[ug_shape, t1_shape](ug, t1)
    else:
        print("[ERROR] Operator not found.")
        res_grad = Tensor[dtype](-1)

    # ug_shape != res_grad.shape(), ug_shape is equal to the result of the forward function not the backward function
    # This will just call elwise_op[add] always
    unbroadcast_add[t1_shape, t1_shape](grad, res_grad)
