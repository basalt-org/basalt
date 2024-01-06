from tensor import Tensor
from random import rand

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.autograd.ops.basics import DOT, ADD



struct Linear:
    """
    A fully connected layer.
    """

    var weights: Node[dtype]
    var bias: Node[dtype]

    fn __init__(inout self, n_input: Int, n_output: Int):
        self.weights = Node[dtype](rand[dtype](n_input, n_output), requires_grad=True, param=True)
        self.bias = Node[dtype](Tensor[dtype](1, n_output), requires_grad=True, param=True)
        GRAPH.add_node(self.weights)
        GRAPH.add_node(self.bias)

    fn forward(inout self, inputs: Node[dtype]) -> Node[dtype]:
        """
        Forward pass of the linear layer.
        """
        # COPY self.weight & self.bias directly from GRAPH
        # Workaround because model parameters are created and change in copies. 
        # TODO: Redo when lifetimes are there. [INVESTIGATE HOW TO AVOID THIS]
        let weights = GRAPH.graph[GRAPH.get_node_idx(self.weights.uuid)]
        let bias = GRAPH.graph[GRAPH.get_node_idx(self.bias.uuid)]

        ######
        # TODO: CHECK IF PARAMS CHANGED !!!!!

        let res = DOT.forward(inputs, weights)
        return ADD.forward(res, bias)

    fn __call__(inout self, inputs: Node[dtype]) -> Node[dtype]:
        return self.forward(inputs)
