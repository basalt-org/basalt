from tensor import Tensor
from random import rand
from math import sqrt

from dainemo import GRAPH
from dainemo.nn.layers import Layer
from dainemo.autograd.node import Node
from dainemo.autograd.ops.basics import DOT, ADD
from dainemo.utils.tensorutils import rand_uniform



struct Linear(Layer):
    """
    A fully connected layer.
    """

    var weights: Node[dtype]
    var bias: Node[dtype]

    fn __init__(inout self, n_input: Int, n_output: Int):
        let k: SIMD[dtype, 1] =  1.0 / n_input
        self.weights = Node[dtype](
            rand_uniform[dtype, nelts](TensorShape(n_input, n_output), -sqrt(k), sqrt(k)),
            requires_grad=True,
            param=True
        )
        self.bias = Node[dtype](
            rand_uniform[dtype, nelts](TensorShape(n_output), -sqrt(k), sqrt(k)),
            requires_grad=True,
            param=True
        )
        GRAPH.add_node(self.weights)
        GRAPH.add_node(self.bias)

    fn forward(self, inputs: Node[dtype]) -> Node[dtype]:
        """
        Forward pass of the linear layer.
        """
        # COPY self.weight & self.bias directly from GRAPH
        # Workaround because model parameters are created and change in copies. 
        # TODO: Redo when lifetimes are there. [INVESTIGATE HOW TO AVOID THIS]
        let weights = GRAPH.graph[GRAPH.get_node_idx(self.weights.uuid)]
        let bias = GRAPH.graph[GRAPH.get_node_idx(self.bias.uuid)]

        let res = DOT.forward(inputs, weights)
        return ADD.forward(res, bias)

    fn __call__(self, inputs: Node[dtype]) -> Node[dtype]:
        return self.forward(inputs)
