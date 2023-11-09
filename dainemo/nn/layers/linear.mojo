from tensor import Tensor
from random import rand
from math import add

from dainemo.autograd.graph import Graph
from dainemo.autograd.ops.basics import Dot
from dainemo.utils.tensorutils import zero, batch_tensor_elwise_op



struct Linear[dtype: DType]:
    '''
    A fully connected layer.
    '''
    
    var n_input: Int
    var n_output: Int
    var weights: Tensor[dtype]
    var bias: Tensor[dtype]

    fn __init__(inout self, n_input: Int, n_output: Int):
        self.n_input = n_input
        self.n_output = n_output
        self.weights = rand[dtype](n_input, n_output)
        self.bias = Tensor[dtype](1, n_output)
        zero[dtype](self.bias)

    @always_inline
    fn forward(inout self, inout graph: Graph[dtype], inputs: Tensor[dtype]) -> Tensor[dtype]:
        '''
        Forward pass of the linear layer.
        '''

        let res = Dot[dtype].forward(graph, inputs, self.weights)

        # TODO: Investigate why bias is not done with the autograd SUM in the "neograd" implementation
        # Does this make bias a constant? / untrainable parameter?
        alias nelts: Int = simdwidthof[dtype]()
        return batch_tensor_elwise_op[dtype, nelts, add](res, self.bias)

    fn __call__(inout self, inout graph: Graph[dtype], inputs: Tensor[dtype]) -> Tensor[dtype]:
        return self.forward(graph, inputs)
