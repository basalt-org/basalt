from random import rand
from tensor import Tensor
from dainemo.utils.uuid import uuid


@value
struct Node[dtype: DType = DType.float32](CollectionElement, Stringable):
    var tensor: Tensor[dtype]
    var requires_grad: Bool
    var uuid: String
    var children: DynamicVector[String]
    var parents: DynamicVector[String]

    fn __init__(inout self, tensor: Tensor[dtype], requires_grad: Bool = False):
        self.tensor = tensor
        self.requires_grad = requires_grad
        self.uuid = uuid()
        self.children = DynamicVector[String]()
        self.parents = DynamicVector[String]()

    fn add_child(inout self, child_id: String):
        self.children.push_back(child_id)

    fn add_parent(inout self, parent_id: String):
        self.parents.push_back(parent_id)

    fn __str__(self) -> String:
        var res = String("Node(")
        res += self.uuid
        res += ")"
        return res



struct Graph[dtype: DType = DType.float32](Stringable):
    var graph: DynamicVector[Node[dtype]]
    var map: DynamicVector[String]

    fn __init__(inout self):
        self.graph = DynamicVector[Node[dtype]]()
        self.map = DynamicVector[String]()

    fn add_node(inout self, inout node: Node[dtype]):

        if self.graph.size != 0:
            var last_node = self.graph[self.graph.size - 1]
            last_node.add_child(node.uuid)
            self.graph[self.graph.size - 1] = last_node     # TODO: remove, waiting for lifetimes (__getitem__ of a dynamic vector returns a copy and not a reference)
            
            node.add_parent(last_node.uuid)

        self.graph.push_back(node)
        self.map.push_back(node.uuid)

    fn __str__(self) -> String:
        var res = String("Graph(")
        for i in range(self.graph.size):
            res += self.graph[i].uuid
            res += " -> "
        res += ")"
        return res


alias dtype = DType.float32
var GRAPH = Graph[dtype]()


fn main():
    
    var t0: Tensor[dtype] = rand[dtype](3, 4)
    var t1: Tensor[dtype] = rand[dtype](1, 4)

    var mynode = Node[dtype](t0, requires_grad=True)
    var mychild = Node[dtype](t1, requires_grad=True)

    GRAPH.add_node(mynode)
    GRAPH.add_node(mychild)

    for i in range(GRAPH.graph.size):
        var n = GRAPH.graph[i]
        print("GRAPH: ", n.uuid)
        for i in range(n.children.size):
            print("\tMYCHILD: ", n.children[i])
        for i in range(n.parents.size):
            print("\tMYPARENT: ", mychild.parents[i])