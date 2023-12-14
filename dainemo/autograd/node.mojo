from math import add, min
from tensor import Tensor, TensorShape

from dainemo import GRAPH
from dainemo.utils.uuid import uuid
from dainemo.utils.tensorutils import fill, elwise_op, tsum


fn backward_fn_placeholder[dtype: DType](
        ug: Tensor[dtype],
        tensor_vec: DynamicVector[String],
        tensor_id: Int
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
    var backward_fn: fn(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]

    var optim_rms_grad: Tensor[dtype]           # TODO: Remove. Only applicable for param == True (extra trait?)
    var optim_momentum_grad: Tensor[dtype]      # TODO: Remove. Only applicable for param == True (extra trait?)

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
        
        self.optim_rms_grad = Tensor[dtype](self.grad.shape())
        self.optim_momentum_grad = Tensor[dtype](self.grad.shape())


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


    fn backward(inout self, upper_grad: Tensor[dtype], retain_graph: Bool = False):
        '''
        Initial entrypoint for the backward pass. (as loss.backward() is called)
        Calculates the order of the backward pass and calls the backward function of the node to calculate the gradients.
        - upper_grad: The gradient to start the backward pass with. Shape should be equal to the shape of the node's tensor.
        - retain_graph: If true, the graph will not reset after the backward pass.
        '''

        if self.requires_grad:
            # TODO: Check if upper_grad.shape == self.tensor.shape (raises)
            self.accumulate_grad(upper_grad)
            
            # 1. Topological sort of the graph.
            # Visit all children so that they aren't included in the backward pass
            # This allows gradient calculation for any intermediate node in the graph
            GRAPH.reset_visited()
            self.visit_all_children()
            var sorted_nodes = DynamicVector[String]()
            self.topological_sort(sorted_nodes)
            GRAPH.reset_visited()

            # 2. Mark as visited & Backward pass on 1st current node without calulating the gradient
            GRAPH.mark_visited(self.uuid)
            self.backward_gradient(retain_graph, calculate_grads=False)
            
            # 3. Calculate the gradients for the nodes in topological order
            for i in range(1, sorted_nodes.size):
                var node = GRAPH.graph[GRAPH.get_node_idx(sorted_nodes[i])]
                GRAPH.mark_visited(node.uuid)
                node.backward_gradient(retain_graph)

            if not retain_graph:
                GRAPH.reset()


    fn backward(inout self, retain_graph: Bool = False):
        '''
        Function overload for: Default upper_grad, tensor of 1.0, with shape equal to the shape of the node's tensor.
        '''
        var upper_grad = Tensor[dtype](self.tensor.shape())
        alias nelts: Int = simdwidthof[dtype]()
        fill[dtype, nelts](upper_grad, 1.0)
        self.backward(upper_grad, retain_graph)


    fn accumulate_grad(inout self, grad: Tensor[dtype]):
        '''
        Accumulates the gradient of the node in the graph.
        '''
        # TODO: self.grad = elwise_op[dtype, nelts, add](self.grad, grad) --> only
        # Lifetimes (__getitem__ of a dynamic vector returns a copy and not a reference)
        let my_idx = GRAPH.get_node_idx(self.uuid)
        var my_node = GRAPH.graph[my_idx]
        # BUG: my_node.grad has type DType.float32 instead of a generic dtype (also see unbroadcast_data)
        # should be: my_node.grad = elwise_op[dtype, nelts, add](my_node.grad, grad)
        for i in range(grad.num_elements()):
            my_node.grad[i] += grad[i].cast[DType.float32]()
        GRAPH.graph[my_idx] = my_node


    fn accumulate_grad2(inout self, grad: Tensor[DType.float32]):
        # BUG: overload as workaround: should be one generic dtype
        let my_idx = GRAPH.get_node_idx(self.uuid)
        var my_node = GRAPH.graph[my_idx]
        alias nelts: Int = simdwidthof[DType.float32]()
        my_node.grad = elwise_op[DType.float32, nelts, add](my_node.grad, grad)
        GRAPH.graph[my_idx] = my_node
    

    fn backward_gradient(inout self, retain_graph: Bool, calculate_grads: Bool = True):
        '''
        Gradient calculation for the node during the backward pass.
        '''

        for c in range(self.children.size):
            let child_idx = GRAPH.get_node_idx(self.children[c])
            let child = GRAPH.graph[child_idx]
            if self.requires_grad and calculate_grads:
                # Identify the index of itself in the child.parents NodeCollection
                # Required when operation has multiple operands to identify the correct gradient function
                var tensor_id: Int = -1
                for t_idx in range(child.parents.size):
                    if child.parents[t_idx] == self.uuid:
                        tensor_id = t_idx
                        break

                var grad = child.backward_fn(child.grad, child.parents, tensor_id)
                self.unbroadcast_data(grad, self.tensor.shape(), child.parent_broadcast_shape)
                self.accumulate_grad2(grad)


    @staticmethod
    fn unbroadcast_data(inout data: Tensor[DType.float32], original_shape: TensorShape, broadcast_shape: TensorShape):
        '''
        Unbroadcasts the data to the original shape of the node.
        '''
        alias none_bc = TensorShape(-1, -1)
        alias nelts: Int = simdwidthof[dtype]()
        if broadcast_shape != none_bc:
            for dim in range(min(original_shape.rank(), broadcast_shape.rank())):
                if original_shape[dim] != broadcast_shape[dim]:
                    data = tsum[DType.float32, 1](data, axis=dim)
      

    fn topological_sort(inout self, inout sorted_nodes: DynamicVector[String]):
        '''
        Topological sort of the graph.
        Efficiently perform the backwards pass by making sure that all the children's gradients are calculated before the parents.
        '''

        # Check if all children are visited
        # 1. If not, topological sort on the children
        if not self.are_children_visited():
            for c in range(self.children.size):
                let idx = GRAPH.get_node_idx(self.children[c])
                var child = GRAPH.graph[idx]
                if not child.visited:
                    child.topological_sort(sorted_nodes)

        # 2. If yes, add node to array 
        #    & topological sort on the parents to go up the graph
        else:
            GRAPH.mark_visited(self.uuid)
            sorted_nodes.push_back(self.uuid)
            for p in range(self.parents.size):
                let idx = GRAPH.get_node_idx(self.parents[p])
                var parent = GRAPH.graph[idx]
                if not parent.visited:
                    parent.topological_sort(sorted_nodes)


    fn reset_relations(inout self):
        '''
        Resets the relations of the node in the graph.
        '''
        self.children = DynamicVector[String]()
        self.parents = DynamicVector[String]()

    
    fn __str__(self) -> String:
        var res = String("Node(")
        res += self.uuid
        res += ")"
        return res