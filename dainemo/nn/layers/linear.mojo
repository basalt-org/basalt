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

    fn __init__(inout self, n_input: Int, n_output: Int):
        self.n_input = n_input
        self.n_output = n_output
        self.weights = Node[dtype](rand[dtype](n_input, n_output), requires_grad=True)
        self.bias = Node[dtype](Tensor[dtype](n_input, n_output), requires_grad=True)

    fn forward(inout self, inout g: Graph[dtype], inputs: Node[dtype]) -> Node[dtype]:
        '''
        Forward pass of the linear layer.
        '''

        let res = DOT[dtype].forward(g, inputs, self.weights)
        return ADD[dtype].forward(g, res, self.bias)

    fn __call__(inout self, inout g: Graph[dtype], inputs: Node[dtype]) -> Node[dtype]:
        return self.forward(g, inputs)
