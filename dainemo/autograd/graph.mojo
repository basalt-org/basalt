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
    var tracking: Bool   ### TODO: --> make this a parameter, compile time known for optimization purposes

    fn __init__(inout self):
        self.graph = GraphNodeCollection[dtype]()
        self.tracking = True


    fn add_edge(inout self, inout result_graph_node: GraphNode[dtype], operand: Node[dtype]):
        '''
        Adds an edge between result node and the corresponding operand node of the operand tensor.
            - Identify the operand graph node in the graph corresponding the the operand node
            - Adds the result node as child to the operand nodes
            - Adds the operand nodes as parents to the result node
        '''
        # 1. Find the operand node in the graph
        var idx = self.get_node(operand)
        if idx == -1:
            # Add operand node to the graph when not found in the collection
            self.add_node(operand)
            idx = self.graph.size - 1

        # 2. Adds the result node as child to the operand nodes
        var operand_graph_node = self.graph.get(idx)
        operand_graph_node.add_child(result_graph_node.node)
        self.graph.replace(idx, operand_graph_node)

        # 3. Adds the operand node as parent to the result graph node
        result_graph_node.add_parent(operand)
    

    fn create_graph_node[backward_fn: String](inout self, result: Tensor[dtype], *operands: Node[dtype]) -> Node[dtype]:
        '''
        To be used in every forward operation and responsible for creating the graph.
        If tracking is enabled it:
            - Creates a GraphNode for the result_tensor & the operands
            - Adds edges to the graphnodes of the the result_tensor & the operands
        '''

        if self.tracking:
            # 1. Create a GraphNode
            '''
            > result_requires_grad:
            Returns True when at least one of the operand nodes requires grad.
            '''
            var requires_grad: Bool = False
            for i in range(operands.__len__()):
                let operand: Node[dtype] = __get_address_as_lvalue(operands[i])
                if operand.requires_grad:
                    requires_grad = True
                    break

            let result_node = Node[dtype](result, requires_grad=requires_grad)
            var result_graph_node = GraphNode[dtype](result_node)

            # TODO: Set backwar_fn & TODO: Set parent_broadcasting_shape
            # The resulting node in the graph always contains it's backward information
            result_graph_node.backward_fn = backward_fn

            # 2. Add edges to the result node & the operands and adds them to the graph
            for i in range(operands.__len__()):
                let operand: Node[dtype] = __get_address_as_lvalue(operands[i])
                self.add_edge(result_graph_node, operand)

            self.graph.append(result_graph_node)
            
            return result_node
        
        else:
            return Node[dtype](result)


    fn add_node(inout self, node: Node[dtype]):
        '''
        Adds a node to the graph.
        '''
        let graph_node = GraphNode[dtype](node)
        self.graph.append(graph_node)


    fn get_node(inout self, node: Node[dtype]) -> Int:
        '''
        Returns the GraphNode (index in the GraphNodeCollection) corresponding to the given node.
        When the node is not found in the graph, returns -1.        
        '''
        
        for i in range(self.graph.size):
            let graph_node = self.graph.get(i)
            if graph_node.node.uuid == node.uuid:
                return i
            
        return -1

    
    fn reset(inout self):
        '''
        Resets the graph.
        '''
        self.graph = GraphNodeCollection[dtype]()


    fn reset_visited(inout self):
        '''
        Marks visited as False for every GraphNode in the graph.
        '''
        for i in range(self.graph.size):
            self.graph.set_visit_value(i, False)

