from tensor import Tensor
from random import rand
from dainemo.utils.tensorutils import zero


struct Linear[dtype: DType]:
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

    fn forward(inout self, inputs: Tensor[dtype]) -> Tensor[dtype]:
        '''
        Forward pass of the linear layer
        '''
        alias nelts: Int = simdwidthof[dtype]()
        return dot[dtype, nelts](inputs, self.weights) + self.bias
