from tensor import Tensor
from random import rand
from dainemo.utils.tensorutils import zero, dot


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
        self.bias = Tensor[dtype](1, n_output)   # batch size?
        zero[dtype](self.bias)

    @always_inline
    fn forward(inout self, inputs: Tensor[dtype]) -> Tensor[dtype]:
        '''
        Forward pass of the linear layer.
        '''
        alias nelts: Int = simdwidthof[dtype]()
        return dot[dtype, nelts](inputs, self.weights) # + self.bias # TODO: + op

    fn __call__(inout self, inputs: Tensor[dtype]) -> Tensor[dtype]:
        return self.forward(inputs)
