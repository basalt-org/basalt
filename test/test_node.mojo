from tensor import Tensor
from random import rand
from testing import assert_true, assert_equal

from dainemo import GRAPH
from dainemo.autograd.node import Node

alias dtype = DType.float32


fn build_test_graph():
    # Building the graph for test purposes
    # A graph with two parent nodes and one child node:
    #       node_1   \
    #                  [-] --> node_3
    #       node_2   / 
    var node_1 = Node(rand[dtype](2, 3))
    var node_2 = Node(rand[dtype](1, 10))
    var node_3 = Node(rand[dtype](2, 2, 2))

    # Define node relationships
    node_1.add_child(node_3)
    node_2.add_child(node_3)
    node_3.add_parent(node_1)
    node_3.add_parent(node_2)

    # Add nodes global graph
    GRAPH.add_node(node_1)
    GRAPH.add_node(node_2)
    GRAPH.add_node(node_3)
    print(GRAPH)


fn test_graph_relations() raises:
    let n1 = GRAPH.graph[0]
    let n2 = GRAPH.graph[1]
    let n3 = GRAPH.graph[2]

    assert_equal(n1.children[0], n3.uuid)    # node_1 -> node_3
    assert_equal(n2.children[0], n3.uuid)    # node_2 -> node_3
    assert_equal(n3.parents[0], n1.uuid)     # node_3 <- node_1
    assert_equal(n3.parents[1], n2.uuid)     # node_3 <- node_2



fn main():

    build_test_graph()

    try:
        assert_equal(GRAPH.graph.size, 3)
        test_graph_relations()
    except:
        print("[ERROR] Error in test_node.py")


    # # visit_all_children & are_children_visited requires a graph
    # var g = Graph[dtype]()
    # g.add_node(node_1)
    # g.add_node(node_2)
    # g.add_node(node_3)


    # print("No children:", node.are_children_visited(g))
    
    # node.add_child(node_2)
    # node.add_child(node_3)

    # print("Unvisited:", node.are_children_visited(g))

    # node.visit_all_children(g)

    # print("Visited:", node.are_children_visited(g))

    # g.reset_visited()

    # print("Graph resets visit:", node.are_children_visited(g))