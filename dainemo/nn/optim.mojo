from tensor import Tensor
from math import mul, add, sqrt, div, sub

from dainemo import GRAPH
from dainemo.utils.tensorutils import elwise_transform, elwise_op, elwise_pow


# <------------Adam------------>
struct Adam:
    var lr: SIMD[dtype, 1]
    var beta1: SIMD[dtype, 1]
    var beta2: SIMD[dtype, 1]
    var epsilon: SIMD[dtype, 1]
    var iter: Int

    fn __init__(inout self, 
                lr: SIMD[dtype, 1] = 0.001, 
                beta1: SIMD[dtype, 1] = 0.9, 
                beta2: SIMD[dtype, 1] = 0.999, 
                epsilon: SIMD[dtype, 1] = 1e-8
            ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0

    fn zero_grad(inout self):
        '''Set all gradients to zero.'''
        GRAPH.zero_grad()

    fn reset(inout self):
        '''Reset the optimizer.'''
        self.iter = 0

    fn step(inout self):
        '''Update parameters.'''
        self.iter += 1

        for idx in range(GRAPH.graph.size):
            var param = GRAPH.graph[idx]

            if param.param and param.requires_grad:
                # 1. Compute adam grads
                param.optim_momentum_grad = elwise_op[dtype, nelts, add](
                    elwise_op[dtype, nelts, mul](self.beta1, param.optim_momentum_grad),
                    elwise_op[dtype, nelts, mul](1 - self.beta1, param.grad)
                )
                param.optim_rms_grad = elwise_op[dtype, nelts, add](
                    elwise_op[dtype, nelts, mul](self.beta2, param.optim_rms_grad),
                    elwise_op[dtype, nelts, mul](1 - self.beta2, elwise_pow[dtype, nelts](param.grad, 2))
                )
                
                # 2. Compute bias-corrected adam grads
                var bias_corrected_momentum_grad = elwise_op[dtype, nelts, div](param.optim_momentum_grad, 1 - (self.beta1 ** self.iter))
                var bias_corrected_rms_grad = elwise_op[dtype, nelts, div](param.optim_rms_grad, 1 - (self.beta2 ** self.iter))
                var delta = elwise_op[dtype, nelts, div](
                    bias_corrected_momentum_grad, 
                    elwise_op[dtype, nelts, add](elwise_transform[dtype, nelts, sqrt](bias_corrected_rms_grad), self.epsilon)
                )

                # 3. Update parameters
                param.tensor = elwise_op[dtype, nelts, sub](param.tensor, elwise_op[dtype, nelts, mul](self.lr, delta))
                GRAPH.graph[idx] = param

                

        



# (possibly also)
# <------------Vanilla Gradient Descent------------>
# <------------Momentum------------>
# <------------RMSProp------------>