
import basalt.nn as nn
from basalt import Graph
from basalt.utils.tensorutils import fill


struct Model[
    g: Graph,
    loss_g: Graph,
    lifetime: MutLifetime,
]:

    var module: nn.Module[g]
    var loss_ref: Reference[nn.Loss[loss_g], __mlir_attr.`1: i1`, lifetime]

    fn __init__(inout self, loss_ref: Reference[nn.Loss[loss_g], __mlir_attr.`1: i1`, lifetime]):
        self.module = nn.Module[g]()
        self.loss_ref = loss_ref

    # TODO
    # fn __init__[loss_g: Graph = Graph()](inout self):
    #     var empty_loss = nn.Loss[loss_g]()
    #     self.__init__(Reference(empty_loss))

    fn forward(inout self, *t_inputs: Tensor[dtype]) -> List[Tensor[dtype]]:
        return self.module.forward(t_inputs ^)

    fn backward(inout self):
        # 1. Zero the gradients
        GRADS.set_zero()

        # 2. Backward loss module, initialize upper gradient with 1.0
        fill(GRADS[loss_g.outputs[0]], 1.0)
        self.loss_ref[].module.backward()

        # 3. Backward module
        self.module.backward()
