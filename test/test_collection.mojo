from random import rand
from tensor import Tensor

from dainemo.autograd.node import Node
from dainemo.utils.collection import NodeCollection
from dainemo.utils.tensorutils import tprint, zero



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

    for i in range(2):
        let res = collection.get(i)
        # print(res.tensor)
        print(res.visited)

    # Example visit nodes
    for i in range(2):
        var res = collection.get(i)
        res.visited = True
        collection.replace(i, res)

    for i in range(2):
        let res = collection.get(i)
        # print(res.tensor)
        print(res.visited)