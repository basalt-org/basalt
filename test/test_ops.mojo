from random import rand
from tensor import Tensor

from dainemo.autograd.ops.basics import DOT, SUM, ADD, SUB, MUL, POW
from dainemo.autograd.graph import Graph
from dainemo.utils.tensorutils import fill


fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()
    var g = Graph[dtype]()

    var t1: Tensor[dtype] = Tensor[dtype](2, 3)
    var t2: Tensor[dtype] = Tensor[dtype](2, 3)
    var t3: Tensor[dtype] = Tensor[dtype](3, 2)
    fill[dtype, nelts](t1, 1.0)
    fill[dtype, nelts](t2, 1.0)
    fill[dtype, nelts](t3, 1.0)


    # <------------ADD------------>
    var res = ADD[dtype].forward(g, t1, t2)
    print(res.tensor)
    print("Graph contains 3 nodes:", g.graph.size)
    g.reset()

    # <------------SUB------------>
    res = SUB[dtype].forward(g, t1, t2)
    print(res.tensor)
    print("Graph contains 3 nodes:", g.graph.size)
    g.reset()

    # <------------MUL------------>
    res = MUL[dtype].forward(g, t1, t2)
    print(res.tensor)
    print("Graph contains 3 nodes:", g.graph.size)
    g.reset()

    res = MUL[dtype].forward(g, t1, 5)
    print(res.tensor)
    print("Graph contains 3 nodes:", g.graph.size)
    g.reset()

    # <------------DOT------------>
    res = DOT[dtype].forward(g, t2, t3)
    print(res.tensor)
    print("Graph contains 3 nodes:", g.graph.size)
    g.reset()

    # <------------POW------------>
    res = POW[dtype].forward(g, t1, 2)
    print(res.tensor)
    print("Graph contains 3 nodes:", g.graph.size)
    g.reset()

    # <------------SUM------------>
    res = SUM[dtype].forward(g, t1, axis=0)
    print(res.tensor)
    print("Graph contains 2 nodes:", g.graph.size)
    g.reset()

    res = SUM[dtype].forward(g, t1, axis=1)
    print(res.tensor)
    print("Graph contains 2 nodes:", g.graph.size)
    g.reset()

    let res_scalar = SUM[dtype].forward(g, t1)
    print(res_scalar.tensor)
    print("Graph contains 2 nodes:", g.graph.size)
    g.reset()