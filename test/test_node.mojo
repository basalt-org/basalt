# from tensor import Tensor
# from random import rand
# from testing import assert_true, assert_equal

# from dainemo import GRAPH
# from dainemo.autograd.node import Node

# alias dtype = DType.float32


# fn build_test_graph():
#     # Building the graph for test purposes
#     # A graph with two parent nodes and one child node:
#     #       node_1   \
#     #                  [-] --> node_3
#     #       node_2   / 
#     var node_1 = Node(rand[dtype](2, 3))
#     var node_2 = Node(rand[dtype](1, 10))
#     var node_3 = Node(rand[dtype](2, 2, 2))

#     # Define node relationships
#     node_1.add_child(node_3)
#     node_2.add_child(node_3)
#     node_3.add_parent(node_1)
#     node_3.add_parent(node_2)

#     # Add nodes global graph
#     GRAPH.add_node(node_1)
#     GRAPH.add_node(node_2)
#     GRAPH.add_node(node_3)
#     print(GRAPH)


# fn test_graph_relations() raises:
#     var n1 = GRAPH.graph[0]
#     var n2 = GRAPH.graph[1]
#     var n3 = GRAPH.graph[2]

#     assert_equal(GRAPH.graph.size, 3)
#     assert_equal(n1.children.size, 1)
#     assert_equal(n2.children.size, 1)
#     assert_equal(n3.children.size, 0)
#     assert_equal(n1.parents.size, 0)
#     assert_equal(n2.parents.size, 0)
#     assert_equal(n3.parents.size, 2)

#     assert_equal(n1.children[0], n3.uuid)    # node_1 -> node_3
#     assert_equal(n2.children[0], n3.uuid)    # node_2 -> node_3
#     assert_equal(n3.parents[0], n1.uuid)     # node_3 <- node_1
#     assert_equal(n3.parents[1], n2.uuid)     # node_3 <- node_2


# fn test_visited_default() raises:
#     var n1 = GRAPH.graph[0]
#     var n2 = GRAPH.graph[1]
#     var n3 = GRAPH.graph[2]

#     assert_equal(n1.visited, False)
#     assert_equal(n2.visited, False)
#     assert_equal(n3.visited, False)
    
#     assert_equal(n1.are_children_visited(), False)
#     assert_equal(n2.are_children_visited(), False)
#     assert_equal(n3.are_children_visited(), True)      # node_3 has no children
#     assert_equal(n1.are_parents_visited(), True)       # node_1 has no parents
#     assert_equal(n2.are_parents_visited(), True)       # node_2 has no parents
#     assert_equal(n3.are_parents_visited(), False)


# fn test_mark_reset_visited() raises:
#     var idx = 0

#     assert_equal(GRAPH.graph[idx].visited, False)
#     GRAPH.mark_visited(GRAPH.graph[idx].uuid)
#     assert_equal(GRAPH.graph[idx].visited, True)
#     GRAPH.reset_visited()
#     assert_equal(GRAPH.graph[idx].visited, False)


# fn test_visit_all_children() raises:
#     var n1 = GRAPH.graph[0]
#     var n2 = GRAPH.graph[1]

#     assert_equal(n1.are_children_visited(), False)
#     n2.visit_all_children()
#     assert_equal(n1.are_children_visited(), True)



# fn main():

#     build_test_graph()

#     try:
#         test_graph_relations()
#         test_visited_default()
#         test_mark_reset_visited()
#         test_visit_all_children()
#     except:
#         print("[ERROR] Error in test_node.py")
