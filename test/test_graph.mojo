from random import rand
from tensor import Tensor

from dainemo.autograd.node import Node, GraphNode
from dainemo.autograd.graph import Graph
from dainemo.autograd.node import backward_fn_placeholder


fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()


    var g = Graph[dtype]()

    let tensor_1 = rand[dtype](2, 10)
    let tensor_2 = rand[dtype](2, 10)
    let tensor_3 = rand[dtype](2, 10)
    let node_1 = Node[dtype](tensor_1)
    let node_2 = Node[dtype](tensor_2)
    let node_3 = Node[dtype](tensor_3)

    # Test get_node
    print("Empty graph, size:", g.graph.size, ", finds no nodes:", g.get_node(node_1))
    g.add_node(node_1)
    g.add_node(node_2)
    print(g.get_node(node_1))
    print(g.get_node(node_2))
    print(g.get_node(node_3))

    # Test add_edge
    var result_node = GraphNode[dtype](node_3)
    
    print("Result node has no parents:", result_node.parents.size)
    print("Node corresponding to tensor_1 has no children:", g.graph.get(g.get_node(node_1)).children.size)

    print("Adding edge from node_1 to result_node")
    g.add_edge(result_node, node_1)

    print("Result node has a parent:", result_node.parents.size)
    print("Node corresponding to tensor_1 has a child:",  g.graph.get(g.get_node(node_1)).children.size)


    # Test create_graph_node
    let tensor_4 = rand[dtype](2, 10)
    print("2 nodes in the graph so far:", g.graph.size)
    
    let node_4 = g.create_graph_node[backward_fn_placeholder[dtype]](tensor_4, node_1, node_2, node_3)
    
    print("4 nodes in the graph:", g.graph.size)
    print("tensor_4 node has 3 parents:", g.graph.get(g.get_node(node_4)).parents.size)