from tensor import Tensor
from algorithm import vectorize, parallelize

from dainemo.autograd.node import Node, GraphNode
from dainemo.utils.collection import NodeCollection, GraphNodeCollection
from dainemo.utils.tensorutils import elwise_op
from math import add



struct Graph[dtype: DType = DType.float32]:
    '''
    Keeps track of all the nodes and its relations in the computational graph.
    Created during the forward pass and used by the backpard pass to 
    compute gradients through autodiff.
    '''

    var graph: GraphNodeCollection[dtype]
    var tracking: Bool                          # TODO: can probably be compile time known
    var parameters: NodeCollection[dtype]       # TODO: shouldn't be part of graph, but of nn.model abstract class
                                                # As Inheritance is not supported yet, temporary solution 
                                                # to store the model parameters in the graph

    fn __init__(inout self):
        self.graph = GraphNodeCollection[dtype]()
        self.tracking = True
        self.parameters = NodeCollection[dtype]()

    fn add_edge(inout self, inout result_graph_node: GraphNode[dtype], operand: Node[dtype]):
        '''
        Adds an edge between result node and the corresponding operand node of the operand tensor.
            - Identify the operand graph node in the graph corresponding the the operand node
            - Adds the result node as child to the operand nodes
            - Adds the operand nodes as parents to the result node.
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
    

    fn create_graph_node[
            backward_fn: fn(ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]
        ](
            inout self, 
            result: Tensor[dtype], 
            *operands: Node[dtype]
        )-> Node[dtype]:
        '''
        To be used in every forward operation and responsible for creating the graph.
        If tracking is enabled it:
            - Creates a GraphNode for the result_tensor & the operands
            - Adds edges to the graphnodes of the the result_tensor & the operands.
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

            # TODO: Set parent_broadcasting_shape
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
        return self.graph.get_idx_by_uuid(node.uuid)

    
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

    
    fn mark_visited(inout self, node: Node[dtype]):
        '''
        Marks the GraphNode corresponding to the given node as visited.
        '''
        let idx = self.get_node(node)
        if idx != -1:
            self.graph.set_visit_value(idx, True)


    fn zero_grad(inout self):
        '''
        Zeros the grad value of every node in the graph & parameters.
        '''
        alias nelts = simdwidthof[dtype]()
        for i in range(self.graph.size):
            self.graph.zero_grad(i)
        for i in range(self.parameters.size):
            self.parameters.zero_grad(i)


    fn update_parameter_grads(inout self, graph_idx: Int):
        # TODO: Can be removed when the lifetime of param nodes are handled correctly.
        # For now: Copies the gradient values of the param nodes in the graph to parameters NodeCollection
        alias nelts: Int = simdwidthof[dtype]()
        let graph_node = self.graph.get(graph_idx)
        if graph_node.node.param:
            let param_idx = self.parameters.get_idx_by_uuid(graph_node.node.uuid)
            if param_idx != -1:
                # Accumulate grad value of the param node in parameters
                let current_grad = self.parameters.get_grad_value(param_idx)
                self.parameters.set_grad_value(param_idx, elwise_op[dtype, nelts, add](current_grad, graph_node.node.grad))
            else:
                print("ERROR: Parameter nodes should be added to (graph.parameters) collection on model creation.")
