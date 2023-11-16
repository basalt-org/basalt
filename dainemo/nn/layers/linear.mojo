from tensor import Tensor
from random import rand
from math import add

from dainemo.autograd.node import Node
from dainemo.autograd.graph import Graph
from dainemo.autograd.ops.basics import DOT, ADD
from dainemo.utils.tensorutils import zero, batch_tensor_elwise_op



struct Linear[dtype: DType]:
    '''
    A fully connected layer.
    '''

    var n_input: Int
    var n_output: Int
    var weights: Node[dtype]
    var bias: Node[dtype]

    fn __init__(inout self, inout g: Graph[dtype], n_input: Int, n_output: Int):
        self.n_input = n_input
        self.n_output = n_output
        self.weights = Node[dtype](rand[dtype](n_input, n_output), requires_grad=True, param=True)
        self.bias = Node[dtype](Tensor[dtype](1, n_output), requires_grad=True, param=True)
        g.parameters.append(self.weights)
        g.parameters.append(self.bias)

    fn forward(inout self, inout g: Graph[dtype], inputs: Node[dtype]) -> Node[dtype]:
        '''
        Forward pass of the linear layer.
        '''
        # Get self.weight & self.bias from g.parameters
        # Workaround because model parameters are created and change in copies. 
        # TODO: Redo when lifetimes are there.
        let weights = g.parameters.get(g.parameters.get_idx_by_uuid(self.weights.uuid))
        let bias = g.parameters.get(g.parameters.get_idx_by_uuid(self.bias.uuid))

        let res = DOT[dtype].forward(g, inputs, weights)
        return ADD[dtype].forward(g, res, bias)

    fn __call__(inout self, inout g: Graph[dtype], inputs: Node[dtype]) -> Node[dtype]:
        return self.forward(g, inputs)
