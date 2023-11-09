from tensor import Tensor
from dainemo.utils.tensorutils import elwise_pow, elwise_op, tsum
from math import sub, mul


# <------------MSE------------>
struct MSELoss[dtype: DType]:
    fn __init__(inout self):
        pass

    fn forward(inout self, outputs: Tensor[dtype], targets: Tensor[dtype]) -> SIMD[dtype, 1]:
        '''Forward pass of MSE.'''
        alias nelts = simdwidthof[dtype]()
        let difference = elwise_op[dtype, nelts, sub](outputs, targets)
        let squared_difference = elwise_pow[dtype, nelts](difference, 2)

        # TODO: Autograd SUM required
        # TODO: The loss function is working but. SUM should be implemented in autograd
        print("MSELoss doesn't support autograd yet")

        let div2N: SIMD[dtype, 1] = (1/(2*outputs.num_elements())).cast[dtype]()
        return div2N * tsum[dtype, nelts](squared_difference)

    fn forward(inout self, output: SIMD[dtype, 1], target: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
        '''Forward pass of MSE on a scalar.'''
        # TODO
        return -1

    fn __call__(inout self, outputs: Tensor[dtype], targets: Tensor[dtype]) -> SIMD[dtype, 1]:
        return self.forward(outputs, targets)

    fn __call__(inout self, outputs: SIMD[dtype, 1], targets: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
        return self.forward(outputs, targets)


# <------------CROSSENTROPY------------>
# TODO

# <------------BINARYCROSSENTROPY------------>
# TODO

# <------------SOFTMAXCROSSENTROPY------------>
# TODO
