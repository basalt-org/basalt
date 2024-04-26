from test_tensorutils import assert_tensors_equal

import basalt
import basalt.nn as nn
from basalt import GRADS
from basalt import Graph, Symbol, OP
from basalt import Tensor, TensorShape
from basalt.autograd.ops.dynamics import CONCAT, SPLIT
from basalt.utils.tensorutils import fill

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


fn create_graph_concat(t1_shape: TensorShape, t2_shape: TensorShape, t3_shape: TensorShape, dim: Int) -> Graph:
    # Testing with 3 operands
    var g = Graph()
    var t1 = g.input(t1_shape, trainable=True)
    var t2 = g.input(t2_shape, trainable=True)
    var t3 = g.input(t3_shape, trainable=True)
    var res = g.concat(t1, t2, t3, dim=dim)
    g.out(res)
    g.loss(res)
    return g ^


fn test_CONCAT_0() raises:
    # default: dim = 0
    # FORWARD
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
    for i in range(4):
        for j in range(2):
            for k in range(3):
                if i < 1: # i because dim = 0
                    expected[i*2*3 + j*3 + k] = 5.0
                elif i >= 1 and i < 2:
                    expected[i*2*3 + j*3 + k] = 10.0
                else:
                    expected[i*2*3 + j*3 + k] = 15.0

    alias graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=0)
    var model = nn.Model[graph]()
    var res = model.forward(t1, t2, t3)
    assert_tensors_equal(res, expected, "almost")

    # BACKWARD
    var ug = Tensor[dtype](4, 2, 3)
    for i in range(4):
        for j in range(2):
            for k in range(3):
                if i < 1: # i because dim = 0
                    ug[i*2*3 + j*3 + k] = 1.0
                elif i >= 1 and i < 2:
                    ug[i*2*3 + j*3 + k] = 2.0
                else:
                    ug[i*2*3 + j*3 + k] = 3.0
    
    model.backward(ug)
    
    var grad1_expected = Tensor[dtype](t1_shape)
    var grad2_expected = Tensor[dtype](t2_shape)
    var grad3_expected = Tensor[dtype](t3_shape)
    fill(grad1_expected, 1.0)
    fill(grad2_expected, 2.0)
    fill(grad3_expected, 3.0)

    # Extracting the gradients
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[0]], grad1_expected, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[1]], grad2_expected, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[2]], grad3_expected, "almost")

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
                if j < 2: # j because dim = 1
                    expected[i*7*5 + j*5 + k] = 5.0
                elif j >= 2 and j < 6:
                    expected[i*7*5 + j*5 + k] = 10.0
                else:
                    expected[i*7*5 + j*5 + k] = 15.0

    alias graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=1)
    var model = nn.Model[graph]()
    var res = model.forward(t1, t2, t3)
    assert_tensors_equal(res, expected, "almost")
    
    # BACKWARD
    var ug = Tensor[dtype](2, 7, 5)
    for i in range(2):
        for j in range(7):
            for k in range(5):
                if j < 2: # j because dim = 1
                    ug[i*7*5 + j*5 + k] = 1.0
                elif j >= 2 and j < 6:
                    ug[i*7*5 + j*5 + k] = 2.0
                else:
                    ug[i*7*5 + j*5 + k] = 3.0
    
    model.backward(ug)
    
    var grad1_expected = Tensor[dtype](t1_shape)
    var grad2_expected = Tensor[dtype](t2_shape)
    var grad3_expected = Tensor[dtype](t3_shape)
    fill(grad1_expected, 1.0)
    fill(grad2_expected, 2.0)
    fill(grad3_expected, 3.0)

    # Extracting the gradients
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[0]], grad1_expected, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[1]], grad2_expected, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[2]], grad3_expected, "almost")
    
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
                if k < 1: # k because dim = 2
                    expected[i*3*6 + j*6 + k] = 5.0
                elif k >= 1 and k < 3:
                    expected[i*3*6 + j*6 + k] = 10.0
                else:
                    expected[i*3*6 + j*6 + k] = 15.0

    alias graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=2)
    var model = nn.Model[graph]()
    var res = model.forward(t1, t2, t3)
    assert_tensors_equal(res, expected, "almost")
    
    # BACKWARD
    var ug = Tensor[dtype](2, 3, 6)
    for i in range(2):
        for j in range(3):
            for k in range(6):
                if k < 1: # k because dim = 2
                    ug[i*3*6 + j*6 + k] = 1.0
                elif k >= 1 and k < 3:
                    ug[i*3*6 + j*6 + k] = 2.0
                else:
                    ug[i*3*6 + j*6 + k] = 3.0
    
    model.backward(ug)
    
    var grad1_expected = Tensor[dtype](t1_shape)
    var grad2_expected = Tensor[dtype](t2_shape)
    var grad3_expected = Tensor[dtype](t3_shape)
    fill(grad1_expected, 1.0)
    fill(grad2_expected, 2.0)
    fill(grad3_expected, 3.0)

    # Extracting the gradients
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[0]], grad1_expected, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[1]], grad2_expected, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[2]], grad3_expected, "almost")

    basalt.reset()


fn main():
    try:
        test_CONCAT_0()
        test_CONCAT_1()
        test_CONCAT_2()
    except e:
        print("[ERROR] Error in dynamic ops")
        print(e)
        return
