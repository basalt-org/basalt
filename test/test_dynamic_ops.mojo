import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import fill

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


fn test_CONCAT() raises:

    # default: dim = 0
    alias t1_shape0 = TensorShape(1, 2, 3)
    alias t2_shape0 = TensorShape(1, 2, 3)
    var t10: Tensor[dtype] = Tensor[dtype](t1_shape0)
    var t20: Tensor[dtype] = Tensor[dtype](t2_shape0)
    fill(t10, 5.0)
    fill(t20, 10.0)

    var expected0 = Tensor[dtype](2, 2, 3)
    for i in range(expected0.num_elements()):
        expected0[i] = 5.0 if i < 1*2*3 else 10.0

    # TODO
    # test_binary_op[OP.CONCAT, t1_shape0, t2_shape0](t10, t20, expected0)
    
    # dim = 1
    alias t1_shape = TensorShape(2, 2, 5)
    alias t2_shape = TensorShape(2, 4, 5)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)

    var expected = Tensor[dtype](2, 6, 5)
    for i in range(2):
        for j in range(6):
            for k in range(5):
                # j < t1_shape[1] because dim=1
                expected[i*6*5 + j*5 + k] = 5.0 if j < t1_shape[1] else 10.0
    
    # TODO
    # test_binary_op[
    #     OP.CONCAT, t1_shape, t2_shape, AttributeVector(Attribute("dim", 1))
    # ](t1, t2, expected)


fn test_backward_CONCAT() raises:
    alias t1_shape = TensorShape(2, 3, 4)
    alias t2_shape = TensorShape(2, 3, 5)
    alias ug_shape = TensorShape(2, 3, 9)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)

    var ug = Tensor[dtype](ug_shape)
    for i in range(2):
        for j in range(3):
            for k in range(9):
                # k < t1_shape[2] because dim=2
                ug[i*3*9 + j*9 + k] = 2.0 if k < t1_shape[2] else 4.0

    var grad1_expected = Tensor[dtype](t1_shape)
    var grad2_expected = Tensor[dtype](t2_shape)
    fill(grad1_expected, 2.0)
    fill(grad2_expected, 4.0)

    # TODO
    # alias attrs = AttributeVector(Attribute("dim", 2))
    # test_binary_op_backward[OP.CONCAT, t1_shape, t2_shape, ug_shape, attrs](t1, t2, ug, grad1_expected, grad2_expected)
    


fn main():
    try:
        test_CONCAT()
    except e:
        print("[ERROR] Error in forward dynamic ops")
        print(e)
        return

    try:
        test_backward_CONCAT()
    except e:
        print("[ERROR] Error in backward dynamic ops")
        print(e)
        return
