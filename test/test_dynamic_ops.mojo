from test_tensorutils import assert_tensors_equal

import basalt
import basalt.nn as nn
from basalt import Graph, Symbol, OP
from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import fill

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


fn create_graph_concat(t1_shape: TensorShape, t2_shape: TensorShape, t3_shape: TensorShape, dim: Int) -> Graph:
    # Testing with 3 operands
    var g = Graph()
    var t1 = g.input(t1_shape)
    var t2 = g.input(t2_shape)
    var t3 = g.input(t3_shape)
    var res = g.concat(t1, t2, t3, dim=dim)
    g.out(res)
    return g ^


fn test_CONCAT_0() raises:
    # default: dim = 0
    alias t1_shape = TensorShape(1, 2, 3)
    alias t2_shape = TensorShape(1, 2, 3)
    alias t3_shape = TensorShape(2, 2, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var t3: Tensor[dtype] = Tensor[dtype](t3_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)
    fill(t3, 15.0)

    var expected = Tensor[dtype](4, 2, 3)
    for i in range(expected.num_elements()):
        if i < 1*2*3:
            expected[i] = 5.0
        elif i >= 1*2*3 and i < 2*2*3:
            expected[i] = 10.0
        else:
            expected[i] = 15.0

    alias graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=0)
    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1, t2, t3)[0]
    assert_tensors_equal(res, expected, "almost")
    basalt.reset()
    

fn test_CONCAT_1() raises:
    # dim = 1
    alias t1_shape = TensorShape(2, 2, 5)
    alias t2_shape = TensorShape(2, 4, 5)
    alias t3_shape = TensorShape(2, 1, 5)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var t3: Tensor[dtype] = Tensor[dtype](t3_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)
    fill(t3, 15.0)

    var expected = Tensor[dtype](2, 7, 5)
    for i in range(2):
        for j in range(7):
            for k in range(5):
                if j < 2: 
                    expected[i*7*5 + j*5 + k] = 5.0
                elif j >= 2 and j < 6:
                    expected[i*7*5 + j*5 + k] = 10.0
                else:
                    expected[i*7*5 + j*5 + k] = 15.0

    alias graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=1)
    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1, t2, t3)[0]
    assert_tensors_equal(res, expected, "almost")
    basalt.reset()


fn test_CONCAT_2() raises:
    # dim = 2
    alias t1_shape = TensorShape(2, 3, 1)
    alias t2_shape = TensorShape(2, 3, 2)
    alias t3_shape = TensorShape(2, 3, 3)
    var t1: Tensor[dtype] = Tensor[dtype](t1_shape)
    var t2: Tensor[dtype] = Tensor[dtype](t2_shape)
    var t3: Tensor[dtype] = Tensor[dtype](t3_shape)
    fill(t1, 5.0)
    fill(t2, 10.0)
    fill(t3, 15.0)

    var expected = Tensor[dtype](2, 3, 6)
    for i in range(2):
        for j in range(3):
            for k in range(6):
                if k < 1: 
                    expected[i*3*6 + j*6 + k] = 5.0
                elif k >= 1 and k < 3:
                    expected[i*3*6 + j*6 + k] = 10.0
                else:
                    expected[i*3*6 + j*6 + k] = 15.0

    alias graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=2)
    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1, t2, t3)[0]
    assert_tensors_equal(res, expected, "almost")
    basalt.reset()


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
        test_CONCAT_0()
        test_CONCAT_1()
        test_CONCAT_2()
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
