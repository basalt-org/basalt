from tensor import Tensor

from dainemo.autograd.node import Node
from dainemo.autograd.ops.basics import SUM, MUL, SUB, POW



# <------------MSE------------>
struct MSELoss:
    fn __init__(inout self):
        pass

    fn forward(inout self, outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        '''Forward pass of MSE.'''

        let difference = SUB.forward(outputs, Node[dtype](targets))
        let squared_difference = POW.forward(difference, 2)
        let res = SUM.forward(squared_difference)
        let div2N: SIMD[dtype, 1] = (1/(2*outputs.tensor.num_elements())).cast[dtype]()
        return MUL.forward(res, div2N)

    fn __call__(inout self, outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        return self.forward(outputs, targets)
        

# <------------CROSSENTROPY------------>
# TODO

# <------------BINARYCROSSENTROPY------------>
# TODO

# <------------SOFTMAXCROSSENTROPY------------>
# TODO
