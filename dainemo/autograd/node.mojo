from math import add
from tensor import Tensor
from algorithm import vectorize, parallelize

from dainemo.autograd.graph import Graph
from dainemo.utils.collection import NodeCollection
from dainemo.utils.uuid import uuid
from dainemo.utils.tensorutils import fill



struct Node[dtype: DType = DType.float32]:
    '''
    A Node data structure that is used to build the computational graph.
        - tensor: The tensor that is stored in the node.
    '''
    
    var tensor: Tensor[dtype]
    var requires_grad: Bool
    var requires_broadcast: Bool
    var uuid: String  # Identifier to find a node by uuid in the graph
    var grad: Tensor[dtype]
    # var grad_fn: Function set in the <ops>.backward() that will be executed during the backward pass.


    fn __init__(inout self, tensor: Tensor[dtype], requires_grad: Bool = False, requires_broadcast: Bool = True):
        self.tensor = tensor
        self.requires_grad = requires_grad              # TODO: can probably compile time known -> parameter
        self.requires_broadcast = requires_broadcast
        self.uuid = uuid()
        self.grad = Tensor[dtype](self.tensor.shape())
        
    fn __copyinit__(inout self, other: Node[dtype]):
        self.tensor = other.tensor
        self.requires_grad = other.requires_grad
        self.requires_broadcast = other.requires_broadcast
        self.uuid = other.uuid
        self.grad = other.grad

    fn backward(inout self, inout g: Graph[dtype], upper_grad: Tensor[dtype], retain_graph: Bool = False):
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
            let idx = g.get_node(self)
            var graph_node = g.graph.get(idx)
            graph_node.backward(g, retain_graph)
            if not retain_graph:
                g.reset()


    fn backward(inout self, inout g: Graph[dtype], retain_graph: Bool = False):
        '''Function overload for: Default upper_grad, a Tensor of 1.0 with shape equal to the shape of the node's tensor.'''
        var upper_grad = Tensor[dtype](self.tensor.shape())
        alias nelts: Int = simdwidthof[dtype]()
        fill[dtype, nelts](upper_grad, 1.0)
        self.backward(g, upper_grad, retain_graph)


    fn accumulate_grad(inout self, grad: Tensor[dtype]):
        '''
        Accumulates the gradient of the node.
        '''
        alias nelts: Int = simdwidthof[dtype]()
        @parameter
        fn vecadd[nelts: Int](idx: Int):
            self.grad.simd_store[nelts](idx, add[dtype, nelts](self.grad.simd_load[nelts](idx), grad.simd_load[nelts](idx)))
        vectorize[nelts, vecadd](self.grad.num_elements())
        
    
    fn backward_gradient(inout self, calculate_grads: Bool = True):
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
    var visited: Bool
    var children: NodeCollection[dtype]
    var parents: NodeCollection[dtype]
    # var parent_broadcast_shape TODO
    var backward_fn: String
    

    fn __init__(inout self, node: Node[dtype]):
        self.node = node
        self.visited = False
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

    fn visit_all_children(inout self, inout g: Graph[dtype]):
        '''
        Marks all children of the node as visited in the graph.
        '''
        for child in self.children:
            let idx = g.get_node(child)
            g.graph.set_visit_value(idx, True)

    fn are_children_visited(inout self, inout g: Graph[dtype]) -> Bool:
        '''
        Checks if all children of the node are visited in the graph.
        '''
        for child in self.children:
            let idx = g.get_node(child)
            if not g.graph.get_visit_value(idx):
                return False
        return True

    fn __copyinit__(inout self, other: GraphNode[dtype]):
        self.node = other.node
        self.visited = other.visited
        self.children = other.children
        self.parents = other.parents
        self.backward_fn = other.backward_fn

    fn backward(inout self, inout g: Graph[dtype], retain_graph: Bool = False):
        '''
        Calculates the order of the backward pass and calls the backward function of the node to calculate the gradients.
        '''

        # 1. Topological sort of the graph.
        # Visit all children so that they aren't included in the backward pass
        # This allows gradient calculation for any intermediate node in the graph
        g.reset_visited()
        self.visit_all_children(g)
        var sorted_nodes = self.topological_sort(g)
        g.reset_visited()

        # 2. Mark as visited & Backward pass on current node without calulating the gradient
        sorted_nodes.remove(0)
        g.mark_visited(self.node)
        

        # 3. Calculate the gradients for the nodes in topological order



    fn topological_sort(inout self, inout g: Graph[dtype]) -> NodeCollection[dtype]:
        '''
        Topological sort of the graph.
        Efficiently perform the backwards pass by making sure that all the children's gradients are calculated before the parents.
        '''
        var sorted_nodes = NodeCollection[dtype]()

        # Check if all children are visited
        # 1. If not, topological sort on the children
        if not self.are_children_visited(g):
            for child in self.children:
                let idx = g.get_node(child)
                var child_graph_node = g.graph.get(idx)
                if not child_graph_node.visited:
                    sorted_nodes += child_graph_node.topological_sort(g)

        # 2. If yes, add node to array 
        #    & topological sort on the parents to go up the graph
        else:
            g.mark_visited(self.node)
            sorted_nodes.append(self.node)
            for parent in self.parents:
                let idx = g.get_node(parent)
                var parent_graph_node = g.graph.get(idx)
                if not parent_graph_node.visited:
                    sorted_nodes += parent_graph_node.topological_sort(g)

        return sorted_nodes