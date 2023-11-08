from math import equal
from tensor import Tensor
from algorithm import vectorize, parallelize

from dainemo.autograd.node import Node, GraphNode
from dainemo.utils.collection import GraphNodeCollection



struct Graph[dtype: DType = DType.float32]:
    '''
    Keeps track of all the nodes and its relations in the computational graph.
    Created during the forward pass and used by the backpard pass to 
    compute gradients through autodiff.
    '''

    var graph: GraphNodeCollection[dtype]
    var tracking: Bool

    fn __init__(inout self):
        self.graph = GraphNodeCollection[dtype]()
        self.tracking = True


    fn add_edge(inout self, inout result_node: GraphNode[dtype], operand: Tensor[dtype]):
        '''
        Adds an edge between result node and the corresponding operand node of the operand tensor.
            - Identify the operand node in the graph
            - Adds a child to the operand node
            - Adds a parent to the result node
        '''
        # 1. Find the operand node in the graph
        var idx = self.get_node(operand)
        if idx == -1:
            # Add operand tensor to the graph when not found in the collection
            self.add_tensor(operand)
            idx = self.graph.size - 1

        # 2. Add child to the operand node in the graph
        var operand_node = self.graph.get(idx)
        operand_node.add_child(result_node.node)
        self.graph.replace(idx, operand_node)

        # 3. Add parent to the result node
        result_node.add_parent(operand_node.node)
    

    fn set_forward_op(inout self, result: Tensor[dtype], *operands: Tensor[dtype]):
        '''
        To be used in every forward operation and responsible for creating the graph.
        If tracking is enabled it:
            - Creates a GraphNode for the result_tensor 
            - Adds edges to the graph from the operands to the result_tensor.
        '''

        if self.tracking:
            # 1. Create a GraphNode
            let node = Node[dtype](result)
            var result_node = GraphNode[dtype](node)

            # TODO: Set backwar_fn & parent_broadcasting_shape

            # 2. Add edges to the graph
            for i in range(operands.__len__()):
                let operand: Tensor[dtype] = __get_address_as_lvalue(operands[i])
                self.add_edge(result_node, operand)

            self.graph.append(result_node)


    fn add_tensor(inout self, tensor: Tensor[dtype]):
        '''
        Adds a tensor to the graph.
        '''
        let node = Node[dtype](tensor)
        let graph_node = GraphNode[dtype](node)
        self.graph.append(graph_node)


    fn get_node(inout self, tensor: Tensor[dtype]) -> Int:
        '''
        Returns the GraphNode (index in the GraphNodeCollection) corresponding to the given tensor.
        When the tensor is not found in the graph, returns -1.        
        '''
        
        # TODO: can probably be parallelized 
        # TODO: revise vectorized elwise_equal, could maybe be stopped early on False
        # TODO: revise is getting the node by tensor is the right way to go

        alias nelts: Int = simdwidthof[dtype]()

        for i in range(self.graph.size):
            let graph_node = self.graph.get(i)
            if not graph_node.node.tensor.shape() == tensor.shape():
                continue
            
            let res = self.elwise_equal[dtype, nelts](graph_node.node.tensor, tensor)
            if res:
                return i
            
        return -1


    @staticmethod
    fn elwise_equal[dtype: DType, nelts: Int](t1: Tensor[dtype], t2: Tensor[dtype]) -> Bool:
        var res: Bool = True
        
        @parameter
        fn vecmath[nelts: Int](idx: Int):
            let t = equal[dtype, nelts](t1.simd_load[nelts](idx), t2.simd_load[nelts](idx))
            if not t.reduce_or():
                res = False
        vectorize[nelts, vecmath](t1.num_elements())
        
        return res
