from math import add, min
from tensor import Tensor, TensorShape

from dainemo import GRAPH
from dainemo.utils.uuid import uuid
from dainemo.utils.tensorutils import fill, elwise_op, tsum


fn backward_fn_placeholder[dtype: DType](
        ug: Tensor[dtype],
        operand_tensors: VariadicListMem[Tensor[dtype]],
        operand_idx: Int
    ) -> Tensor[dtype]:
    print("[ERROR]: Backward function placeholder")
    return Tensor[dtype](ug.shape())


@value
struct Node[dtype: DType = DType.float32](CollectionElement, Stringable):
    '''
    A Node data structure that is used to build the computational graph.
        - tensor: The tensor that is stored in the node.
        - requires_grad: If true, the node will be included in the backward pass.
        - requires_broadcast: TODO
        - uuid: Identifier to find a node by uuid in the graph
        - grad: The gradient of the node.
        - param: If true, the node is a parameter of the model. (requires_grad indicates if it is trainable or not).
        - children: The children of the node in the graph.
        - parents: The parents of the node in the graph.
    '''
    
    var tensor: Tensor[dtype]
    var requires_grad: Bool                     # TODO: can probably be compile time known
    var requires_broadcast: Bool
    var uuid: String
    var grad: Tensor[dtype]
    var param: Bool                             # TODO: can probably be compile time known
    var visited: Bool
    var children: DynamicVector[String]
    var parents: DynamicVector[String]
    var parent_broadcast_shape: TensorShape
    var backward_fn: fn(ug: Tensor[dtype], operand_tensors: VariadicListMem[Tensor[dtype]], operand_idx: Int) -> Tensor[dtype]

    # var optim_rms_grad: Tensor[dtype]           # TODO: Remove. Etra value for the optimizer to avoid code duplication in absence of inheritance & lifetimes
    # var optim_momentum_grad: Tensor[dtype]      # TODO: Remove. Etra value for the optimizer to avoid code duplication in absence of inheritance & lifetimes

    fn __init__(inout self, tensor: Tensor[dtype], requires_grad: Bool = False, requires_broadcast: Bool = True, param: Bool = False):
        self.tensor = tensor
        self.requires_grad = requires_grad
        self.requires_broadcast = requires_broadcast
        self.uuid = uuid()
        self.grad = Tensor[dtype](self.tensor.shape())
        self.param = param
        self.visited = False
        self.children = DynamicVector[String]()
        self.parents = DynamicVector[String]()
        self.parent_broadcast_shape = tensor.shape()
        self.backward_fn = backward_fn_placeholder[dtype]
        
        # self.optim_rms_grad = Tensor[dtype](self.grad.shape())
        # self.optim_momentum_grad = Tensor[dtype](self.grad.shape())
        # self.optim_momentum_grad = Tensor[dtype](self.grad.shape())

    fn add_child(inout self, node: Node[dtype]):
        ''''
        Adds a child to the node.
        '''
        self.children.push_back(node.uuid)
    
    fn add_parent(inout self, node: Node[dtype]):
        '''
        Adds a parent to the node.
        '''
        self.parents.push_back(node.uuid)

    fn visit_all_children(inout self):
        '''
        Marks all children of the node as visited in the graph.
        '''
        for i in range(self.children.size):
            GRAPH.mark_visited(self.children[i])
    
    fn are_children_visited(inout self) -> Bool:
        '''
        Checks if all children of the node are visited in the graph.
        '''
        for i in range(self.children.size):
            let idx = GRAPH.get_node_idx(self.children[i])
            if not GRAPH.graph[idx].visited:
                return False
        return True

    fn are_parents_visited(inout self) -> Bool:
        '''
        Checks if all parents of the node are visited in the graph.
        '''
        for i in range(self.parents.size):
            let idx = GRAPH.get_node_idx(self.parents[i])
            if not GRAPH.graph[idx].visited:
                return False
        return True



    # fn backward(inout self, inout g: Graph[dtype], upper_grad: Tensor[dtype], retain_graph: Bool = False):
    #     '''
    #     Initial entrypoint for the backward pass. (as loss.backward() is called)
    #     Initializes the backward pass by calling the backward function of the corresponding graph_node.
    #     Which is aware if its children and parents in the graph.
    #     - upper_grad: The gradient to start the backward pass with. Shape should be equal to the shape of the node's tensor.
    #     - retain_graph: If true, the graph will not reset after the backward pass.
    #     '''

    #     if self.requires_grad:
    #         # TODO: Check if upper_grad.shape == self.tensor.shape (raises)
    #         let idx = g.get_node(self)
    #         var graph_node = g.graph.get(idx)
    #         self.accumulate_grad(g, idx, upper_grad)
    #         graph_node.backward(g, retain_graph)
    #         if not retain_graph:
    #             g.reset()


    # fn backward(inout self, inout g: Graph[dtype], retain_graph: Bool = False):
    #     '''
    #     Function overload for: Default upper_grad, a Tensor of 1.0 with shape equal to the shape of the node's tensor.
    #     '''
    #     var upper_grad = Tensor[dtype](self.tensor.shape())
    #     alias nelts: Int = simdwidthof[dtype]()
    #     fill[dtype, nelts](upper_grad, 1.0)
    #     self.backward(g, upper_grad, retain_graph)


    # fn accumulate_grad(inout self, inout g: Graph[dtype], idx: Int, grad: Tensor[dtype]):
    #     '''
    #     Accumulates the gradient of the node in the graph.
    #     '''
    #     let current_grad = g.graph.get_grad_value(idx)
    #     alias nelts: Int = simdwidthof[dtype]()
    #     self.grad = elwise_op[dtype, nelts, add](current_grad, grad)
    #     g.graph.set_grad_value(idx, self.grad)

    
    # fn backward_gradient(inout self, inout g: Graph[dtype], retain_graph: Bool, calculate_grads: Bool = True):
    #     '''
    #     Gradient calculation for the node during the backward pass.
    #     '''
        
    #     let idx = g.get_node(self)
    #     var graph_node = g.graph.get(idx)

    #     for child in graph_node.children:
    #         let child_idx = g.get_node(child)
    #         var child_graph_node = g.graph.get(child_idx)
    #         if self.requires_grad and calculate_grads:
                
    #             # Identify the index of itself in the child.parents NodeCollection
    #             # Required when operation has multiple operands to identify the correct gradient function
    #             let node_id = child_graph_node.parents.get_idx_by_uuid(self.uuid)
    #             let upper_grad = g.graph.get_grad_value(child_idx)
    #             var grad = child_graph_node.backward_fn(upper_grad, child_graph_node.parents, node_id)
    #             self.unbroadcast_data(grad, self.tensor.shape(), child_graph_node.parent_broadcast_shape)
    #             self.accumulate_grad(g, idx, grad)

    #         if not retain_graph and child_graph_node.are_parents_visited(g):
    #             #TODO: update can be removed when copies are avoided
    #             g.update_parameter_grads(child_idx) # Share the grads with the params of the optimizer 
    #             g.graph.remove(child_idx)
    #     if not retain_graph and graph_node.are_parents_visited(g):
    #         #TODO: update can be removed when copies are avoided
    #         g.update_parameter_grads(g.get_node(self)) # Share the grads with the params of the optimizer
    #         g.graph.remove(g.get_node(self))

    # @staticmethod
    # fn unbroadcast_data(inout data: Tensor[dtype], original_shape: TensorShape, broadcast_shape: TensorShape):
    #     '''
    #     Unbroadcasts the data to the original shape of the node.
    #     '''
    #     alias none_bc = TensorShape(-1, -1)
    #     alias nelts: Int = simdwidthof[dtype]()
    #     if broadcast_shape != none_bc:
    #         for dim in range(min(original_shape.rank(), broadcast_shape.rank())):
    #             if original_shape[dim] != broadcast_shape[dim]:
    #                 data = tsum[dtype, 1](data, axis=dim)

    fn __str__(self) -> String:
        var res = String("Node(")
        res += self.uuid
        res += ")"
        return res
            






# struct GraphNode[dtype: DType = DType.float32]:
#     '''
#     A Node in the computational graph.
#     Monitors the relation between all the incoming edges (=parents) and the outgoing edges (=children).
#     '''
#     var node: Node[dtype]
#     var visited: Bool
#     var children: NodeCollection[dtype]
#     var parents: NodeCollection[dtype]
#     var parent_broadcast_shape: TensorShape
#     var backward_fn: fn(ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]
    

#     fn __init__(inout self, node: Node[dtype]):
#         self.node = node
#         self.visited = False
#         self.children = NodeCollection[dtype]()
#         self.parents = NodeCollection[dtype]()
#         self.parent_broadcast_shape = node.tensor.shape()
#         self.backward_fn = backward_fn_placeholder[dtype]
        





    # fn backward(inout self, inout g: Graph[dtype], retain_graph: Bool = False):
    #     '''
    #     Calculates the order of the backward pass and calls the backward function of the node to calculate the gradients.
    #     '''

    #     # 1. Topological sort of the graph.
    #     # Visit all children so that they aren't included in the backward pass
    #     # This allows gradient calculation for any intermediate node in the graph
    #     g.reset_visited()
    #     self.visit_all_children(g)
    #     var sorted_nodes = self.topological_sort(g)
    #     g.reset_visited()

    #     # 2. Mark as visited & Backward pass on 1st current node without calulating the gradient
    #     sorted_nodes.remove(0)
    #     g.mark_visited(self.node)
    #     self.node.backward_gradient(g, retain_graph, calculate_grads=False)
        
    #     # 3. Calculate the gradients for the nodes in topological order
    #     for node in sorted_nodes:
    #         g.mark_visited(node)
    #         node.backward_gradient(g, retain_graph)
            

    # fn topological_sort(inout self, inout g: Graph[dtype]) -> NodeCollection[dtype]:
    #     '''
    #     Topological sort of the graph.
    #     Efficiently perform the backwards pass by making sure that all the children's gradients are calculated before the parents.
    #     '''
    #     #TODO: Sorted nodes is a copy where the uuid is used to fecth the node from the graph
    #     # Could be more efficient is when this is a collection of pointers to the graph nodes
    #     var sorted_nodes = NodeCollection[dtype]()

    #     # Check if all children are visited
    #     # 1. If not, topological sort on the children
    #     if not self.are_children_visited(g):
    #         for child in self.children:
    #             let idx = g.get_node(child)
    #             var child_graph_node = g.graph.get(idx)
    #             if not child_graph_node.visited:
    #                 sorted_nodes += child_graph_node.topological_sort(g)

    #     # 2. If yes, add node to array 
    #     #    & topological sort on the parents to go up the graph
    #     else:
    #         g.mark_visited(self.node)
    #         sorted_nodes.append(self.node)
    #         for parent in self.parents:
    #             let idx = g.get_node(parent)
    #             var parent_graph_node = g.graph.get(idx)
    #             if not parent_graph_node.visited:
    #                 sorted_nodes += parent_graph_node.topological_sort(g)

    #     return sorted_nodes