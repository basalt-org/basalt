from tensor import Tensor
from random import rand

from dainemo.autograd.node import Node, GraphNode
from dainemo.utils.uuid import uuid





fn main():

    # Test uuid
    print(uuid())
    print(uuid())
    print(uuid())


    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()
    
    let node_1: Node[dtype] = Node(rand[dtype](2, 3))
    let node_2: Node[dtype] = Node(rand[dtype](1, 10))
    let node_3: Node[dtype] = Node(rand[dtype](2, 2, 2))

    var node = GraphNode[dtype](node_1)

    print("No children:", node.are_children_visited())
    
    node.add_child(node_2)
    node.add_child(node_3)

    print("Unvisited:", node.are_children_visited())

    node.visit_all_children()

    print("Visited:", node.are_children_visited())