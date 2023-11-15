from tensor import Tensor
from random import rand
from dainemo.nn.optim import Adam
from dainemo.autograd.graph import Graph, Node





fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()

    var optim = Adam[dtype](lr=0.001)
    var g = Graph[dtype]()

    let tensor_1 = rand[dtype](2, 10)
    let tensor_2 = rand[dtype](2, 10)
    let tensor_3 = rand[dtype](2, 10)
    let node_1 = Node[dtype](tensor_1)
    let node_2 = Node[dtype](tensor_2)
    let node_3 = Node[dtype](tensor_3)

    g.add_node(node_1)
    g.add_node(node_2)

    # TODO