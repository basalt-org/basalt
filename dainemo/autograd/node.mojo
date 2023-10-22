from tensor import Tensor



struct Node[dtype: DType]:
    '''
    A node in the computational graph.
    Monitors the relation between all the incoming edges (=parents) and the outgoing edges (=children).
    '''

    # var ptr: Pointer[Self]
    var tensor: Tensor[dtype]
    var children: DynamicVector[Pointer[Node[dtype]]]
    var parents: DynamicVector[Pointer[Node[dtype]]]
    var visited: Bool

    def __init__(inout self, tensor: Tensor[dtype]):
        self.tensor = tensor
        self.children = DynamicVector[Pointer[Self]]()
        self.parents = DynamicVector[Pointer[Self]]()
        self.visited = False

    #     self.ptr = Self.get_null()
    
    # def add_child(inout self, other: Node[dtype]):
    #     ''''
    #     Adds a child to the node.
    #     '''
    #     self.children.push_back(other.ptr.get_null())


    # def add_parent(inout self, other: Node[dtype]):
    #     '''
    #     Adds a parent to the node.
    #     '''
    #     self.parents.push_back(other.ptr.get_null())