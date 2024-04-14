
import basalt.nn as nn
from basalt import Graph
from basalt.utils.tensorutils import fill



struct Model[
    g: Graph,
]:
    var module: nn.Module[g]

    fn __init__(inout self):
        self.module = nn.Module[g]()

    fn forward(inout self, *t_inputs: Tensor[dtype]) -> List[Tensor[dtype]]:
        return self.module.forward(t_inputs ^)

    fn backward[
        loss_g: Graph,
        lifetime: MutLifetime,
    ](inout self, loss: Reference[nn.Loss[loss_g], __mlir_attr.`1: i1`, lifetime]):

        # 1. Zero the gradients
        loss[].module.grads.set_zero()
        self.module.grads.set_zero() 

        # 2. Backward loss module
        loss[].backward()

        # 3. Backward module (given loss uppergrad)
        var upper_grad = loss[].module.grads[g.outputs[0]] # TODO; generalize for multiple outputs
        self.module.backward(upper_grad ^)
        # NOTE might be missing something here
        # # TODO: could be removed if all tensors were initiated in a global collection
        # for i in range(g.outputs.size):
        #     ...


        
