from random import rand
from tensor import Tensor

from dainemo.autograd.node import Node, GraphNode
from dainemo.autograd.graph import Graph



fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()


    var g = Graph[dtype]()

    let tensor_1 = rand[dtype](2, 10)
    let tensor_2 = rand[dtype](2, 10)
    let tensor_3 = rand[dtype](2, 10)

    
    # Test elwise_equal 
    let res_true = g.elwise_equal[dtype, nelts](tensor_1, tensor_1)
    print("Expected True:", res_true)

    let res_false = g.elwise_equal[dtype, nelts](tensor_1, tensor_2)
    print("Expected False:", res_false)


    # Test get_node
    print("Empty graph, size:", g.graph.size, ", finds no nodes:", g.get_node(tensor_1))
    g.add_tensor(tensor_1)
    g.add_tensor(tensor_2)
    print(g.get_node(tensor_1))
    print(g.get_node(tensor_2))
    print(g.get_node(tensor_3))


    # Test add_edge
    let node = Node[dtype](tensor_3)
    var result_node = GraphNode[dtype](node)
    
    print("Result node has no parents:", result_node.parents.size)
    print("Node corresponding to tensor_1 has no children:", g.graph.get(g.get_node(tensor_1)).children.size)

    g.add_edge(result_node, tensor_1)

    print("Result node has a parent:", result_node.parents.size)
    print("Node corresponding to tensor_1 has a child:",  g.graph.get(g.get_node(tensor_1)).children.size)


    # Test set_forward_op
    let tensor_4 = rand[dtype](2, 10)
    print("2 nodes in the graph so far:", g.graph.size)
    
    g.set_forward_op(tensor_4, tensor_3, tensor_2, tensor_1)
    
    print("4 nodes in the graph:", g.graph.size)
    print("tensor_4 node has 3 parents:", g.graph.get(g.get_node(tensor_4)).parents.size)