from random import rand
from tensor import Tensor
from testing import assert_equal

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.autograd.node import backward_fn_placeholder

alias dtype = DType.float32


fn build_test_graph():
    # Building the graph for test purposes
    # A graph with two parent nodes and one child node:
    #       node_1   \
    #                  [-] --> node_3
    #       node_2   / 
    let node_1 = Node(rand[dtype](1, 10))
    let node_2 = Node(rand[dtype](1, 10))
    let tensor_3 = rand[dtype](1, 10)

    # Define node relationships using operation [-] with:
    #      result: tensor_3
    #      operands: node_1, node_2
    _ = GRAPH.create_graph_node[backward_fn_placeholder[dtype]](tensor_3, node_1, node_2)
    print(GRAPH)


fn test_graph_relations() raises:
    let n1 = GRAPH.graph[0]
    let n2 = GRAPH.graph[1]
    let n3 = GRAPH.graph[2]

    assert_equal(GRAPH.graph.size, 3)
    assert_equal(n1.children.size, 1)
    assert_equal(n2.children.size, 1)
    assert_equal(n3.children.size, 0)
    assert_equal(n1.parents.size, 0)
    assert_equal(n2.parents.size, 0)
    assert_equal(n3.parents.size, 2)

    assert_equal(n1.children[0], n3.uuid)    # node_1 -> node_3
    assert_equal(n2.children[0], n3.uuid)    # node_2 -> node_3
    assert_equal(n3.parents[0], n1.uuid)     # node_3 <- node_1
    assert_equal(n3.parents[1], n2.uuid)     # node_3 <- node_2


fn main():

    build_test_graph()

    try:
        test_graph_relations()
    except:
        print("[ERROR] Error in graph.py")
