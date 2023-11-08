from tensor import Tensor

from dainemo.utils.collection import NodeCollection


struct Node[dtype: DType = DType.float32]:
    var tensor: Tensor[dtype]
    var visited: Bool

    fn __init__(inout self, tensor: Tensor[dtype]):
        self.tensor = tensor
        self.visited = False

    fn __copyinit__(inout self, other: Node[dtype]):
        self.tensor = other.tensor
        self.visited = other.visited    


struct GraphNode[dtype: DType = DType.float32]:
    '''
    A node in the computational graph.
    Monitors the relation between all the incoming edges (=parents) and the outgoing edges (=children).
    '''
    var node: Node[dtype]
    var children: NodeCollection[dtype]
    var parents: NodeCollection[dtype]

    fn __init__(inout self, node: Node[dtype]):
        self.node = node
        self.children = NodeCollection[dtype]()
        self.parents = NodeCollection[dtype]()

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