from tensor import Tensor
from dainemo.autograd.graph import Graph
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

    fn set_adam_grads(inout self, inout g: Graph[dtype], param_idx: Int, rms_grads: Tensor[dtype], momentum_grads: Tensor[dtype]):
        # Set the adam grads of the parameter nodes. Remove when Lifetimes are added.
        var current_param = g.parameters.get(param_idx)
        current_param.optim_rms_grad = rms_grads
        current_param.optim_momentum_grad = momentum_grads
        g.parameters.replace(param_idx, current_param)

    fn step(inout self, inout g: Graph[dtype]):
        '''Update parameters.'''
        
        for param in g.parameters:
            print(param.grad)
            print(param.optim_momentum_grad)
            print(param.optim_rms_grad)
            print('----------------------------------')
        
        self.iter += 1



# (possibly also)
# <------------Vanilla Gradient Descent------------>
# <------------Momentum------------>
# <------------RMSProp------------>