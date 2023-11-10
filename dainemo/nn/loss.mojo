from tensor import Tensor
from math import sub, mul

from dainemo.autograd.node import Node
from dainemo.autograd.graph import Graph
from dainemo.autograd.ops.basics import SUM, MUL, SUB, POW
from dainemo.utils.tensorutils import elwise_pow, elwise_op, tsum



# <------------MSE------------>
struct MSELoss[dtype: DType]:
    fn __init__(inout self):
        pass

    fn forward(inout self, inout g: Graph[dtype], outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        '''Forward pass of MSE.'''

        let difference = SUB[dtype].forward(g, outputs, Node[dtype](targets))
        let squared_difference = POW[dtype].forward(g, difference, 2)
        let res = SUM[dtype].forward(g, squared_difference)
        let div2N: SIMD[dtype, 1] = (1/(2*outputs.tensor.num_elements())).cast[dtype]()
        return MUL[dtype].forward(g, res, div2N)

    fn __call__(inout self, inout g: Graph[dtype], outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        return self.forward(g, outputs, targets)
        

# <------------CROSSENTROPY------------>
# TODO

# <------------BINARYCROSSENTROPY------------>
# TODO

# <------------SOFTMAXCROSSENTROPY------------>
# TODO
