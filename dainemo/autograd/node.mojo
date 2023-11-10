from tensor import Tensor

from dainemo.utils.collection import NodeCollection
from dainemo.utils.uuid import uuid



struct Node[dtype: DType = DType.float32]:
    '''
    A Node data structure that is used to build the computational graph.
        - tensor: The tensor that is stored in the node.
        - visited: A flag that is used to mark the node as visited during the backward pass.
    '''
    
    var tensor: Tensor[dtype]
    var visited: Bool
    var requires_grad: Bool
    var requires_broadcast: Bool
    var uuid: String  # Identifier to find a node by uuid in the graph
    # var grad: Tensor[dtype]
    # var grad_fn: Function set in the <ops>.backward() that will be executed during the backward pass.


    fn __init__(inout self, tensor: Tensor[dtype], requires_grad: Bool = False, requires_broadcast: Bool = True):
        self.tensor = tensor
        self.visited = False
        self.requires_grad = requires_grad
        self.requires_broadcast = requires_broadcast
        self.uuid = uuid()
        
    fn __copyinit__(inout self, other: Node[dtype]):
        self.tensor = other.tensor
        self.visited = other.visited
        self.requires_grad = other.requires_grad
        self.requires_broadcast = other.requires_broadcast
        self.uuid = other.uuid


struct GraphNode[dtype: DType = DType.float32]:
    '''
    A Node in the computational graph.
    Monitors the relation between all the incoming edges (=parents) and the outgoing edges (=children).
    '''
    var node: Node[dtype]
    var children: NodeCollection[dtype]
    var parents: NodeCollection[dtype]
    # var parent_broadcast_shape TODO
    var backward_fn: String
    

    fn __init__(inout self, node: Node[dtype]):
        self.node = node
        self.children = NodeCollection[dtype]()
        self.parents = NodeCollection[dtype]()
        self.backward_fn = "None"
        

    fn add_child(inout self, node: Node[dtype]):
        ''''
        Adds a child to the node.
        '''
        self.children.append(node)
    
    fn add_parent(inout self, node: Node[dtype]):
        '''
        Adds a parent to the node.
        '''
        self.parents.append(node)

    fn visit_all_children(inout self):
        '''
        Marks all children of the node as visited.
        '''
        for i in range(self.children.size):
            var child = self.children.get(i)
            child.visited = True
            self.children.replace(i, child)

    fn are_children_visited(inout self) -> Bool:
        '''
        Checks if all children of the node are visited.
        '''
        for child in self.children:
            if not child.visited:
                return False
        return True

    fn __copyinit__(inout self, other: GraphNode[dtype]):
        self.node = other.node
        self.children = other.children
        self.parents = other.parents
        self.backward_fn = other.backward_fn