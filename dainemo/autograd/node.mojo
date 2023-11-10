from math import add
from tensor import Tensor
from algorithm import vectorize, parallelize

from dainemo.autograd.graph import Graph
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
    var grad: Tensor[dtype]
    # var grad_fn: Function set in the <ops>.backward() that will be executed during the backward pass.


    fn __init__(inout self, tensor: Tensor[dtype], requires_grad: Bool = False, requires_broadcast: Bool = True):
        self.tensor = tensor
        self.visited = False
        self.requires_grad = requires_grad              # TODO: can probably compile time known -> parameter
        self.requires_broadcast = requires_broadcast
        self.uuid = uuid()
        self.grad = Tensor[dtype](self.tensor.shape())
        
    fn __copyinit__(inout self, other: Node[dtype]):
        self.tensor = other.tensor
        self.visited = other.visited
        self.requires_grad = other.requires_grad
        self.requires_broadcast = other.requires_broadcast
        self.uuid = other.uuid

    fn backward(inout self, inout graph: Graph[dtype], upper_grad: Tensor[dtype], retain_graph: Bool = False):
        '''
        Initial entrypoint for the backward pass. (as loss.backward() is called)
        Initializes the backward pass by calling the backward function of the corresponding graph_node.
        Which is aware if its children and parents in the graph.
        - upper_grad: The gradient to start the backward pass with. Shape should be equal to the shape of the node's tensor.
        - retain_graph: If true, the graph will not reset after the backward pass.
        '''

        if self.requires_grad:
            # TODO: Check if upper_grad.shape == self.tensor.shape (raises)
            self.accumulate_grad(upper_grad)
            let idx = graph.get_node(self)
            let graph_node = graph.graph.get(idx)
            graph_node.backward(retain_graph)
            if not retain_graph:
                graph.reset()


    fn accumulate_grad(inout self, grad: Tensor[dtype]):
        '''
        Accumulates the gradient of the node.
        '''
        alias nelts: Int = simdwidthof[dtype]()
        @parameter
        fn vecadd[nelts: Int](idx: Int):
            self.grad.simd_store[nelts](idx, add[dtype, nelts](self.grad.simd_load[nelts](idx), grad.simd_load[nelts](idx)))
        vectorize[nelts, vecadd](self.grad.num_elements())
        
    
    fn calculate_gradient(inout self, calculate_grads: Bool = True):
        '''
        Gradient calculation for the node during the backward pass.
        '''
        
        print("calculating gradient of node: " + self.uuid)
        #TODO
        pass
        

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

    fn backward(inout self, retain_graph: Bool = False):
        '''
        Calculates the order of the backward pass and calls the backward function of the node to calculate the gradients.
        '''

        # 1. Visit all children so that they aren't included in the backward pass
        # Allows gradient calculation for any intermediate node in the graph

        # self.visit_all_children()
        pass