from tensor import Tensor
from math import sub, mul

from dainemo.autograd.graph import Graph
from dainemo.autograd.ops.basics import SUM
from dainemo.utils.tensorutils import elwise_pow, elwise_op, tsum



# <------------MSE------------>
struct MSELoss[dtype: DType]:
    fn __init__(inout self):
        pass

    fn forward(inout self, inout g: Graph[dtype], outputs: Tensor[dtype], targets: Tensor[dtype]) -> SIMD[dtype, 1]:
        '''Forward pass of MSE.'''
        alias nelts = simdwidthof[dtype]()
        let difference = elwise_op[dtype, nelts, sub](outputs, targets)
        let squared_difference = elwise_pow[dtype, nelts](difference, 2)
        
        let div2N: SIMD[dtype, 1] = (1/(2*outputs.num_elements())).cast[dtype]()
        
        return div2N * SUM[dtype].forward(g, squared_difference)


    fn __call__(inout self, inout g: Graph[dtype], outputs: Tensor[dtype], targets: Tensor[dtype]) -> SIMD[dtype, 1]:
        return self.forward(g, outputs, targets)


# <------------CROSSENTROPY------------>
# TODO

# <------------BINARYCROSSENTROPY------------>
# TODO

# <------------SOFTMAXCROSSENTROPY------------>
# TODO
