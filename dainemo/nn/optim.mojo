from math import add, mul, div, sqrt, sub
from algorithm import vectorize, parallelize

from .model import Parameters, calc_n_tensors

from dainemo import Graph
from dainemo.utils.collection import Collection
from dainemo.utils.string_dict import StringDict
from dainemo.utils.tensorutils import elwise_op, elwise_pow, elwise_transform



struct Adam[g: Graph, N: Int = calc_n_tensors(g)]:
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

        self.rms_grads = Collection(N)
        self.momentum_grads = Collection(N)

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

        # Loop over all tensor that require_grad = True (i.e. keys in grad_map)
        @parameter
        fn p_step(i: Int):
            var param_tensor_id = parameters.params_map.get(
                parameters.grads_map.keys[i], -1
            )
            var tensor_id = parameters.grads_map.get(parameters.grads_map.keys[i], -1)

            # TODO
            # Investigate most efficient implementation of the Adam optimizer
            # Inplace updates, state values optim_momentum_grad/optim_rms_grad required?

            # 1. Compute adam grads

            # NOTE: Might require the Adam struct to include two more collections
            #   - optim_momentum_grad for each parameter
            #   - optim_rms_grad for each parameter

            # 2. Compute bias-corrected adam grads

            # 3. Update model parameters

            # We should be able to do this operations in a more clean way in the future I think. Like maybe all this operations could be a graph?

            var grads_shape = __get_address_as_lvalue(
                parameters.grads.offset(tensor_id).address
            ).shape()

            var momentum_grads_address = self.momentum_grads.offset(
                tensor_id
            ).address
            var rms_grads_address = self.rms_grads.offset(tensor_id).address
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

            vectorize[v_step, nelts](grads_shape.num_elements())
        
        parallelize[p_step](len(parameters.grads_map.keys), len(parameters.grads_map.keys))

    fn allocate_rms_and_momentum(inout self, inout parameters: Parameters):
        # They are initialized to zero
        # Loop over all tensor that require_grad = True (i.e. inside model.parameters.grads)
        for i in range(parameters.grads.size):
            self.rms_grads.append(
                Tensor[dtype](
                    __get_address_as_lvalue(parameters.grads.offset(i).address).shape()
                )
            )
            self.momentum_grads.append(
                Tensor[dtype](
                    __get_address_as_lvalue(parameters.grads.offset(i).address).shape()
                )
            )
