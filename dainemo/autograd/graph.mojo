from tensor import Tensor, TensorShape
from algorithm import vectorize, parallelize

from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import elwise_op, zero
from math import add, max



struct Graph[dtype: DType = DType.float32, tracking: Bool = True](Stringable):
    '''
    Keeps track of all the nodes and its relations in the computational graph.
    Created during the forward pass and used by the backpard pass to 
    compute gradients through autodiff.
    '''

    var keys: DynamicVector[String]
    var graph: DynamicVector[Node[dtype]]


    fn __init__(inout self):
        self.keys = DynamicVector[String]()
        self.graph = DynamicVector[Node[dtype]]()


    fn add_edge(inout self, inout result_node: Node[dtype], operands: VariadicListMem[Node[dtype]]):
        '''
        Adds an edge between result node and the corresponding operand node of the operand tensor.
            - Identify the operand node in the graph corresponding to the operand node
            - Adds the result node as child to the operand nodes
            - Adds the operand nodes as parents to the result node.
        '''
        for operand_ptr in operands:
            var operand: Node[dtype] = __get_address_as_lvalue(operand_ptr)

            # 1. Find the operand node in the graph
            var idx = self.get_node_idx(operand.uuid)
            if idx == -1:
                # Add operand node to the graph when not found
                self.add_node(operand)
                idx = self.graph.size - 1

            # 2. Adds the result node as child to the operand
            # TODO: self.graph[idx].add_child(result_node)
            # Lifetimes (__getitem__ of a dynamic vector returns a copy and not a reference)
            operand = self.graph[idx]
            operand.add_child(result_node)
            self.graph[idx] = operand

            # 3. Adds the operand node as parent to the result graph node
            result_node.add_parent(operand)
    
        self.add_node(result_node)


    fn create_graph_node[
            backward_fn: fn(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]
        ](
            inout self, 
            result: Tensor[dtype],
            *operands: Node[dtype]
        ) -> Node[dtype]:
        '''
        To be used in every forward operation and responsible for creating the graph.
        If tracking is enabled it:
            - Creates a Node in the computaitonal graph for the result_tensor & the operands
            - Sets the backward_fn & parent_broadcast_shape of the result_node
            - Adds edges to the graphnodes of the the result_tensor & the operands.
        '''

        if tracking:
            # 1. Create a Node from the resulting tensor
            var result_node = Node[dtype](result, requires_grad=self.result_requires_grad(operands))

            # 2. The resulting node in the graph always contains it's backward information
            result_node.backward_fn = backward_fn
            result_node.parent_broadcast_shape = self.get_broadcasting_shape(operands, result)       

            # 3. Add edges to the result node & the operands and adds them to the graph
            self.add_edge(result_node, operands)
            
            return result_node
        
        else:
            return Node[dtype](result)


    @staticmethod
    fn result_requires_grad(operands: VariadicListMem[Node[dtype]]) -> Bool:
        '''
        Returns True when at least one of the operand nodes requires grad.
        '''
        for operand_ptr in operands:
            if __get_address_as_lvalue(operand_ptr).requires_grad:
                return True
        return False


    @staticmethod
    fn get_broadcasting_shape(operands: VariadicListMem[Node[dtype]], result: Tensor[dtype]) -> TensorShape:
        '''
        Broadcast multiple shapes to find a common compatible shape using only loops.
        Returns the broadcast shape of the given operands.
        '''
        # TODO: REFACTOR, Only supports rank 2 operands for now.
        # from testing import assert_true
        let max_rank: Int = 2

        alias none_bc = TensorShape(-1, -1)
        var bc_shape = DynamicVector[Int](max_rank)
        bc_shape.push_back(-1)
        bc_shape.push_back(-1)
        for i in range(max_rank):
            var current_max: Int = 1
            for operand_ptr in operands:
                let operand: Node[dtype] = __get_address_as_lvalue(operand_ptr)
                # _ = assert_true(operand.tensor.rank() <= 2, "Broadcasting only supports up to rank 2 tensors.")
                let operand_shape = operand.tensor.shape()
                if i < operand_shape.rank():
                    let dim_size = operand_shape[max_rank - i - 1]
                    if dim_size > 1:
                        if current_max != 1 and current_max != dim_size:
                            # Broadcasting not supported for given operands.
                            return none_bc
                        current_max = dim_size
            bc_shape[max_rank - i - 1] = current_max

        let broadcast_shape = TensorShape(bc_shape)
        return broadcast_shape


    fn add_node(inout self, inout node: Node[dtype]):
        '''
        Adds a node to the graph.
        '''
        self.keys.push_back(node.uuid)
        self.graph.push_back(node)


    fn get_node_idx(inout self, node_uuid: String) -> Int:
        '''
        Returns the index of the corresponding node in the graph.
        When the node is not found in the graph, returns -1.
        '''
        for i in range(self.keys.size):
            if self.keys[i] == node_uuid:
                return i
        return -1

    
    fn reset(inout self):
        '''
        Resets the graph.
        Except for the trainable parameters.
        '''
        var param_keys = DynamicVector[String]()
        var param_graph = DynamicVector[Node[dtype]]()
        for idx in range(self.keys.size):
            if self.graph[idx].param:
                param_keys.push_back(self.keys[idx])
                param_graph.push_back(self.graph[idx])

        self.keys = param_keys
        self.graph = param_graph


    fn reset_visited(inout self):
        '''
        Marks visited as False for every Node in the graph.
        '''
        for idx in range(self.graph.size):
            # TODO: self.graph[idx].visited = False 
            # Lifetimes (__getitem__ of a dynamic vector returns a copy and not a reference)
            var node = self.graph[idx]
            node.visited = False
            self.graph[idx] = node

    
    fn mark_visited(inout self, node_uuid: String):
        '''
        Marks the Node corresponding to the given uuid as visited in the graph.
        '''
        let idx = self.get_node_idx(node_uuid)
        if idx != -1:
            # TODO: self.graph[idx].visited = True 
            # Lifetimes (__getitem__ of a dynamic vector returns a copy and not a reference)
            var node = self.graph[idx]
            node.visited = True
            self.graph[idx] = node


    fn zero_grad(inout self):
        '''
        Zeros the grad value of every node in the graph & parameters.
        '''
        for idx in range(self.graph.size):
            # TODO: zero[dtype](self.graph[idx].grad)  --> only
            # Lifetimes (__getitem__ of a dynamic vector returns a copy and not a reference)
            var node = self.graph[idx]
            zero[dtype](node.grad)
            self.graph[idx] = node


    fn __str__(self) -> String:
        var res = String("Graph[\n")
        for i in range(self.graph.size):
            res += "\t" + self.graph[i].__str__() + "\n"
        res += "]"
        return res