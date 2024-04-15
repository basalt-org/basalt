from math import add, mul, div, sqrt, sub
from algorithm import vectorize, parallelize

from .module import Parameters

from basalt import TENSORS, GRADS
from basalt import Graph, Tensor, TensorShape
from basalt.utils.collection import Collection


fn get_trainable_parameters(g: Graph) -> List[Symbol]:
    """
    Get all symbols of trainable parameters.
    """

    var trainable_parameters = List[Symbol]()

    for i in range(len(g.params)):
        if g.params.symbols[i].trainable:
            trainable_parameters.append(g.params.symbols[i])

    return trainable_parameters ^


struct Adam[
    g: Graph,
    trainable_parameters: List[Symbol] = get_trainable_parameters(g),
]:

    var lr: SIMD[dtype, 1]
    var beta1: SIMD[dtype, 1]
    var beta2: SIMD[dtype, 1]
    var epsilon: SIMD[dtype, 1]
    var iter: Int

    var rms_grads: Collection
    var momentum_grads: Collection

    fn __init__(
        inout self,
        lr: SIMD[dtype, 1] = 0.001,
        beta1: SIMD[dtype, 1] = 0.9,
        beta2: SIMD[dtype, 1] = 0.999,
        epsilon: SIMD[dtype, 1] = 1e-8,
    ):
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0

        # Capacity of the collections should be the n of trainable parameters
        self.rms_grads = Collection(capacity=len(trainable_parameters))
        self.momentum_grads = Collection(capacity=len(trainable_parameters))
        
        self.allocate_rms_and_momentum()

    fn step(inout self):
        """Update model parameters."""
        self.iter += 1

        # Loop over all trainable parameters
        @parameter
        fn p_step(i: Int):
            var param = trainable_parameters[i]

            @parameter
            fn v_step[nelts: Int](j: Int):
                var momentum_grads = self.momentum_grads[param].load[nelts](j)
                var rms_grads = self.rms_grads[param].load[nelts](j)
                var grads = GRADS[param].load[nelts](j)
                var params = TENSORS[param].load[nelts](j)

                # Momentum beta 1
                # f1 = beta1 * momentum + (1 - beta1) * grad
                momentum_grads = self.beta1 * momentum_grads + (1 - self.beta1) * grads
                self.momentum_grads[param].store[nelts](j, momentum_grads)

                # Bias correction
                # f2 = f1 / (1 - beta1 ** iter)
                momentum_grads = momentum_grads / (1 - self.beta1**self.iter)

                # RMS beta 2
                # f1 = beta2 * rms + (1 - beta2) * grad ** 2
                rms_grads = self.beta2 * rms_grads + (1 - self.beta2) * grads * grads
                self.rms_grads[param].store[nelts](j, rms_grads)

                # Bias correction
                # f2 = f1 / (1 - beta2 ** iter)
                rms_grads = rms_grads / (1 - self.beta2**self.iter)

                # tensor = tensor - lr * (f2 / (sqrt(rms) + epsilon))
                params = params - self.lr * (
                    momentum_grads / (sqrt(rms_grads) + self.epsilon)
                )
                TENSORS[param].store[nelts](j, params)

            vectorize[v_step, 1](param.shape.num_elements())

        parallelize[p_step](len(trainable_parameters))

    fn allocate_rms_and_momentum(inout self):
        # They are initialized to zero
        # Loop over all trainable parameters
        for i in range(len(trainable_parameters)):
            var param = trainable_parameters[i]
            self.rms_grads.append(Tensor[dtype](param.shape), param)
            self.momentum_grads.append(Tensor[dtype](param.shape), param)
