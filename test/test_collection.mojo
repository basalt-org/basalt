from random import rand
from tensor import Tensor

from dainemo.autograd.node import Node, GraphNode
from dainemo.utils.collection import NodeCollection, GraphNodeCollection



fn main():

    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()
    
    let node_1: Node[dtype] = Node(rand[dtype](2, 3))
    let node_2: Node[dtype] = Node(rand[dtype](1, 10))
    let node_3: Node[dtype] = Node(rand[dtype](2, 2, 2))
    
    var collection = NodeCollection[dtype]()

    collection.append(node_1)
    print("Collection size (expected 1): ", collection.size, "capacity (expected 1): ", collection.capacity)
    
    collection.append(node_2)
    print("Collection size (expected 2): ", collection.size, "capacity (expected 2): ", collection.capacity)

    collection.append(node_3)
    print("Collection size (expected 3): ", collection.size, "capacity (expected 4): ", collection.capacity)


    var graph = GraphNodeCollection[dtype]()
    graph.append(GraphNode[dtype](node_1))
    graph.append(GraphNode[dtype](node_2))
    graph.append(GraphNode[dtype](node_3))

    for graph_node in graph:
        print(graph_node.visited)

    # Example visit first 2 nodes
    # By idx
    graph.set_visit_value(0, True)
    graph.set_visit_value(1, True)

    for graph_node in graph:
        print(graph_node.visited)

    # Copy collection    
    print("Copying collection")
    var graph_copy = graph.copy()
    for graph_node in graph_copy:
        print(graph_node.visited)

    # Adding two collections
    print("Adding two collections")
    var added = collection + collection.copy()
    for node in added:
        print(node.uuid)

    print("iadding two collections")
    collection += collection.copy()
    for node in collection:
        print(node.uuid)

    # Removing nodes
    print("Removing nodes")
    collection.remove(5)
    collection.remove(3)
    collection.remove(1)
    for node in collection:
        print(node.uuid)
    print(collection.capacity)

    # Removing nodes and freeing up memory
    collection.remove(0)
    for node in collection:
        print(node.uuid)
    print(collection.capacity)

    