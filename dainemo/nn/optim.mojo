from tensor import Tensor
from dainemo.autograd.graph import Graph
from dainemo.utils.tensorutils import elwise_transform, elwise_op, elwise_pow
from math import mul, add, sqrt, div, sub
'''Optimizers.'''


# <------------Adam------------>
struct Adam[dtype: DType]:

    # var params #TODO: Waiting for lifetimes. Storing references of parameter nodes. (Get them from model class)
    # When the optimizer has the params variable, Remove from graph. & Remove update_parameter_grads from Node.backward_gradient
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

    fn zero_grad(inout self, inout g: Graph[dtype]):
        '''Set all gradients to zero.'''
        g.zero_grad()

    fn reset(inout self):
        '''Reset the optimizer.'''
        self.iter = 0

    fn step(inout self, inout g: Graph[dtype]):
        '''Update parameters.'''
        alias nelts = simdwidthof[dtype]()
        self.iter += 1

        for param_idx in range(g.parameters.size):
            var param = g.parameters.get(param_idx)

            if param.requires_grad:
                
                # 1. Compute adam grads
                param.optim_momentum_grad = elwise_op[dtype, nelts, add](
                    elwise_op[dtype, nelts, mul](self.beta1, param.optim_momentum_grad),
                    elwise_op[dtype, nelts, mul](1 - self.beta1, param.grad)
                )
                param.optim_rms_grad = elwise_op[dtype, nelts, add](
                    elwise_op[dtype, nelts, mul](self.beta2, param.optim_rms_grad),
                    elwise_op[dtype, nelts, mul](1 - self.beta2, elwise_transform[dtype, nelts, sqrt](param.grad))
                )

                # 2. Compute bias-corrected adam grads
                let bias_corrected_momentum_grad = elwise_op[dtype, nelts, div](param.optim_momentum_grad, 1 - (self.beta1 ** self.iter))
                let bias_corrected_rms_grad = elwise_op[dtype, nelts, div](param.optim_rms_grad, 1 - (self.beta2 ** self.iter))
                let delta = elwise_op[dtype, nelts, div](
                    bias_corrected_momentum_grad, 
                    elwise_op[dtype, nelts, add](elwise_transform[dtype, nelts, sqrt](bias_corrected_rms_grad), self.epsilon)
                )

                # 3. Update parameters
                param.tensor = elwise_op[dtype, nelts, sub](param.tensor, elwise_op[dtype, nelts, mul](self.lr, delta))
                g.parameters.replace(param_idx, param)

                

        



# (possibly also)
# <------------Vanilla Gradient Descent------------>
# <------------Momentum------------>
# <------------RMSProp------------>