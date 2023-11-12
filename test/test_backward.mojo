from random import rand
from tensor import Tensor
from math import equal
from algorithm import vectorize, parallelize
from testing import assert_true, assert_equal

from dainemo.autograd.ops.basics import DOT, SUM, ADD, SUB, MUL, POW
from dainemo.autograd.graph import Graph
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import fill


fn elwise_equal[dtype: DType, nelts: Int](t1: Tensor[dtype], t2: Tensor[dtype]) -> Bool:
    var res: Bool = True
    
    @parameter
    fn vecmath[nelts: Int](idx: Int):
        let t = equal[dtype, nelts](t1.simd_load[nelts](idx), t2.simd_load[nelts](idx))
        if not t.reduce_or():
            res = False
    vectorize[nelts, vecmath](t1.num_elements())
    
    return res


fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()
    var g = Graph[dtype]()

    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var t3: Tensor[dtype] = Tensor[dtype](3, 2)
    var upper_grad: Tensor[dtype] = Tensor[dtype](2, 3)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 2.0)
    fill[dtype, nelts](t3, 3.0)
    fill[dtype, nelts](upper_grad, 1.0)


    # <------------ADD------------>
    var res = ADD[dtype].forward(g, t1, t2)
    
    # By construction: The id of the nodes in the graph should be 0, 1, 2
    let g_t1_id: Int = 0
    let g_t2_id: Int = 1
    let res_id: Int = 2
    # Verify
    for gnode in g.graph:
        print(elwise_equal[dtype, nelts](gnode.node.tensor, t1), elwise_equal[dtype, nelts](gnode.node.tensor, t2), elwise_equal[dtype, nelts](gnode.node.tensor, res.tensor))
    
    var gn = g.graph.get(res_id)            # The graph node of the result
    _ = assert_equal(gn.parents.size, 2)    # Should have 2 parents
    
    # Get id of the operand nodes in the gn.parents collection
    var t1_id = gn.parents.get_idx_by_uuid(g.graph.get(g_t1_id).node.uuid)
    var t2_id = gn.parents.get_idx_by_uuid(g.graph.get(g_t2_id).node.uuid)
    _ = assert_equal(t1_id, 0)
    _ = assert_equal(t2_id, 1)

    var ug1 = gn.backward_fn(upper_grad, gn.parents, t1_id)
    var ug2 = gn.backward_fn(upper_grad, gn.parents, t2_id)
    
    _ = assert_true(elwise_equal[dtype, nelts](ug1, upper_grad))
    _ = assert_true(elwise_equal[dtype, nelts](ug2, upper_grad))
    
    g.reset()


    # <------------SUB------------>
    res = SUB[dtype].forward(g, t1, t2)
    gn = g.graph.get(2)
    t1_id = 0
    t2_id = 1

    ug1 = gn.backward_fn(upper_grad, gn.parents, t1_id)
    ug2 = gn.backward_fn(upper_grad, gn.parents, t2_id)

    var expected_ug1 = upper_grad
    var expected_ug2 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug2, -1.0)
    
    _ = assert_true(elwise_equal[dtype, nelts](ug1, upper_grad))
    _ = assert_true(elwise_equal[dtype, nelts](ug2, expected_ug2))
    g.reset()


    # <------------MUL------------>
    res = MUL[dtype].forward(g, t1, t2)
    gn = g.graph.get(2)
    t1_id = 0
    t2_id = 1

    ug1 = gn.backward_fn(upper_grad, gn.parents, t1_id)
    ug2 = gn.backward_fn(upper_grad, gn.parents, t2_id)

    expected_ug1 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug1, 2.0)
    expected_ug2 = Tensor[dtype](2, 3)
    fill[dtype, nelts](expected_ug2, 1.0)

    _ = assert_true(elwise_equal[dtype, nelts](ug1, expected_ug1))
    _ = assert_true(elwise_equal[dtype, nelts](ug2, expected_ug2))