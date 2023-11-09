from random import rand
from tensor import Tensor

from dainemo.autograd.ops.basics import DOT, SUM
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


    # <------------DOT------------>
    var res = DOT[dtype].forward(g, t2, t3)
    print(res)
    print("Graph contains 3 nodes:", g.graph.size)
    g.reset()

    # <------------SUM------------>
    res = SUM[dtype, axis=0].forward(g, t1)
    print(res)
    print("Graph contains 2 nodes:", g.graph.size)
    g.reset()

    res = SUM[dtype, axis=1].forward(g, t1)
    print(res)
    print("Graph contains 2 nodes:", g.graph.size)
    g.reset()