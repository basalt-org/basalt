from math import add, mul, div, sqrt, sub
from algorithm import vectorize, parallelize

from .model import Parameters, collect_trainable_parameters

from basalt import Graph
from basalt.utils.collection import Collection



fn get_num_trainable_parameters[g: Graph]() -> Int:
    var count = 0
    for i in range(len(g.params)):
        if g.params.symbols[i].trainable:
            count += 1
    return count



struct Adam[g: Graph]:
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
        # TODO: len(model.parameters.trainable_parameters) when model parameters are passed as reference.
        var N = get_num_trainable_parameters[g]()  
        self.rms_grads = Collection(capacity=N)
        self.momentum_grads = Collection(capacity=N)

    fn zero_grad(inout self, inout parameters: Parameters):
        """Set all gradients to zero."""
        for grad_idx in range(parameters.grads_map.count):
            memset_zero[dtype](
                __get_address_as_lvalue(
                    parameters.grads.offset(
                        parameters.grads_map.values[grad_idx]
                    ).address
                ).data(),
                __get_address_as_lvalue(
                    parameters.grads.offset(
                        parameters.grads_map.values[grad_idx]
                    ).address
                ).num_elements(),
            )

    fn step(inout self, inout parameters: Parameters):
        """Update model parameters."""
        self.iter += 1

        # Loop over all updatable parameters that require_grad = True (i.e. keys in grad_map)
        @parameter
        fn p_step(i: Int):
            var param_tensor_id = parameters.params_map.get(
                parameters.trainable_parameters[i], -1
            )
            var tensor_id = parameters.grads_map.get(parameters.trainable_parameters[i], -1)
            var grads_shape = __get_address_as_lvalue(
                parameters.grads.offset(tensor_id).address
            ).shape()

            var momentum_grads_address = self.momentum_grads.offset(i).address
            var rms_grads_address = self.rms_grads.offset(i).address
            var grads_address = parameters.grads.offset(tensor_id).address
            var params_address = parameters.params.offset(param_tensor_id).address

            @parameter
            fn v_step[nelts: Int](j: Int):
                # f1 = beta1 * momentum + (1 - beta1) * grad
                # f2 = f1 / (1 - beta1 ** iter)
                # tensor = tensor - lr * (f2 / (sqrt(rms) + epsilon))

                var momentum_grads = __get_address_as_lvalue(momentum_grads_address).simd_load[nelts](j)
                var rms_grads = __get_address_as_lvalue(rms_grads_address).simd_load[nelts](j)
                var grads = __get_address_as_lvalue(grads_address).simd_load[nelts](j)
                var params = __get_address_as_lvalue(params_address).simd_load[nelts](j)

                # Momentum beta 1
                # f1 = beta1 * momentum + (1 - beta1) * grad
                momentum_grads = self.beta1 * momentum_grads + (1 - self.beta1) * grads
                __get_address_as_lvalue(momentum_grads_address).simd_store[nelts](j, momentum_grads)
                # Bias correction
                # f2 = f1 / (1 - beta1 ** iter)
                momentum_grads = momentum_grads / (1 - self.beta1 ** self.iter)

                # RMS beta 2
                # f1 = beta2 * rms + (1 - beta2) * grad ** 2
                rms_grads = self.beta2 * rms_grads + (1 - self.beta2) * grads * grads
                __get_address_as_lvalue(rms_grads_address).simd_store[nelts](j, rms_grads)
                # Bias correction
                # f2 = f1 / (1 - beta2 ** iter)
                rms_grads = rms_grads / (1 - self.beta2 ** self.iter)

                params = params - self.lr * (momentum_grads / (sqrt(rms_grads) + self.epsilon))


                __get_address_as_lvalue(params_address).simd_store[nelts](j, params)

            vectorize[v_step, 1](grads_shape.num_elements())
        
        parallelize[p_step](len(parameters.trainable_parameters), len(parameters.trainable_parameters))

    fn allocate_rms_and_momentum(inout self, inout parameters: Parameters):
        # They are initialized to zero
        # Loop over all updatable parameters that require_grad = True (i.e. inside model.parameters.trainable_parameters)
        for i in range(len(parameters.trainable_parameters)):   
            var tensor_id = parameters.grads_map.get(parameters.trainable_parameters[i], -1)

            self.rms_grads.append(
                Tensor[dtype](
                    __get_address_as_lvalue(parameters.grads.offset(tensor_id).address).shape()
                )
            )
            self.momentum_grads.append(
                Tensor[dtype](
                    __get_address_as_lvalue(parameters.grads.offset(tensor_id).address).shape()
                )
            )
