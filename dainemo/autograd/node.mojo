from math import add
from tensor import Tensor

from dainemo.autograd.graph import Graph
from dainemo.utils.collection import NodeCollection
from dainemo.utils.uuid import uuid
from dainemo.utils.tensorutils import fill, elwise_op



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
    # var param: Bool   # TODO: Mark node as param to save weights & biasses & to pass on grads to optimizer 

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
            let idx = g.get_node(self)
            var graph_node = g.graph.get(idx)
            self.accumulate_grad(g, idx, upper_grad)
            graph_node.backward(g, retain_graph)
            if not retain_graph:
                g.reset()


    fn backward(inout self, inout g: Graph[dtype], retain_graph: Bool = False):
        '''
        Function overload for: Default upper_grad, a Tensor of 1.0 with shape equal to the shape of the node's tensor.
        '''
        var upper_grad = Tensor[dtype](self.tensor.shape())
        alias nelts: Int = simdwidthof[dtype]()
        fill[dtype, nelts](upper_grad, 1.0)
        self.backward(g, upper_grad, retain_graph)


    fn accumulate_grad(inout self, inout g: Graph[dtype], idx: Int, grad: Tensor[dtype]):
        '''
        Accumulates the gradient of the node in the graph.
        '''
        let current_grad = g.graph.get_grad_value(idx)
        alias nelts: Int = simdwidthof[dtype]()
        self.grad = elwise_op[dtype, nelts, add](current_grad, grad)
        g.graph.set_grad_value(idx, self.grad)

    
    fn backward_gradient(inout self, inout g: Graph[dtype], retain_graph: Bool, calculate_grads: Bool = True):
        '''
        Gradient calculation for the node during the backward pass.
        '''
        
        let idx = g.get_node(self)
        var graph_node = g.graph.get(idx)

        for child in graph_node.children:
            let child_idx = g.get_node(child)
            var child_graph_node = g.graph.get(child_idx)
            if self.requires_grad and calculate_grads:
                
                # Identify the index of itself in the child.parents NodeCollection
                # Required when operation has multiple operands to identify the correct gradient function
                let node_id = child_graph_node.parents.get_idx_by_uuid(self.uuid)
                let upper_grad = g.graph.get_grad_value(child_idx)
                let grad = child_graph_node.backward_fn(upper_grad, child_graph_node.parents, node_id)
                # TODO: Broadcasting
                self.accumulate_grad(g, idx, grad)

                # TODO: Before removing the nodes in the graph, find a way to share the grads with the params of the optimizer

            if not retain_graph and child_graph_node.are_parents_visited(g):
                g.graph.remove(child_idx)
        if not retain_graph and graph_node.are_parents_visited(g):
            g.graph.remove(g.get_node(self))


fn backward_fn_placeholder[dtype: DType](ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]:
    # TODO: Error when called (raises)
    print("Backward function placeholder")
    return Tensor[dtype](ug.shape())


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
    var backward_fn: fn(ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]
    

    fn __init__(inout self, node: Node[dtype]):
        self.node = node
        self.visited = False
        self.children = NodeCollection[dtype]()
        self.parents = NodeCollection[dtype]()
        self.backward_fn = backward_fn_placeholder[dtype]
        

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

    fn are_parents_visited(inout self, inout g: Graph[dtype]) -> Bool:
        '''
        Checks if all parents of the node are visited in the graph.
        '''
        for parent in self.parents:
            let idx = g.get_node(parent)
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

        # 2. Mark as visited & Backward pass on 1st current node without calulating the gradient
        sorted_nodes.remove(0)
        g.mark_visited(self.node)
        self.node.backward_gradient(g, retain_graph, calculate_grads=False)
        
        # 3. Calculate the gradients for the nodes in topological order
        for node in sorted_nodes:
            g.mark_visited(node)
            node.backward_gradient(g, retain_graph)
            

    fn topological_sort(inout self, inout g: Graph[dtype]) -> NodeCollection[dtype]:
        '''
        Topological sort of the graph.
        Efficiently perform the backwards pass by making sure that all the children's gradients are calculated before the parents.
        '''
        #TODO: Sorted nodes is a copy where the uuid is used to fecth the node from the graph
        # Could be more efficient is when this is a collection of pointers to the graph nodes
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