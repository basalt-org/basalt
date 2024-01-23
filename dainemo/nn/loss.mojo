from tensor import Tensor
from math import add
from math.limit import max_finite

from dainemo.autograd.node import Node
from dainemo.autograd.ops.basics import SUM, MUL, SUB, POW, LOG, ADD
from dainemo.utils.tensorutils import elwise_op


# <------------MSE------------>
struct MSELoss:
    fn __init__(inout self):
        pass

    fn forward(inout self, outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        """
        Forward pass of MSE.
        """
        # 1/2N * sum( (outputs - targets)^2 )

        let difference = SUB.forward(outputs, Node[dtype](targets))
        let squared_difference = POW.forward(difference, 2)
        let res = SUM.forward(squared_difference)
        let div2N: SIMD[dtype, 1] = (1/(2*outputs.tensor.num_elements())).cast[dtype]()
        return MUL.forward(res, div2N)

    fn __call__(inout self, outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        return self.forward(outputs, targets)


# <------------CROSSENTROPY------------>
struct CrossEntropyLoss:
    fn __init__(inout self):
        pass

    fn forward(inout self, inout outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        """
        Forward pass of CrossEntropy.
        Epsilons is a small value for numerical stability to prevent log(0).
        """
        # -1/N * sum( targets * log(outputs + epsilon) )

        alias epsilon = 1e-9
        let add_eps = ADD.forward(outputs, epsilon)

        let logout = LOG.forward(outputs)
        let targets_logout = MUL.forward(Node[dtype](targets), logout)
        let entropy = SUM.forward(targets_logout)
        let negDivN: SIMD[dtype, 1] = (-1/outputs.tensor.num_elements()).cast[dtype]()
        return MUL.forward(entropy, negDivN)

    fn __call__(inout self, inout outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        return self.forward(outputs, targets)


# <------------BINARYCROSSENTROPY------------>
struct BCELoss:
    fn __init__(inout self):
        pass

    fn forward(inout self, inout outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        """
        Forward pass of BCE.
        Epsilons is a small value for numerical stability to prevent log(0).
        """
        # -1/N * sum( targets * log(outputs+epsilon) + (1-targets) * log(1-outputs+epsilon) )

        alias epsilon = 1e-9
        outputs.tensor = elwise_op[dtype, nelts, add](outputs.tensor, epsilon)
        var tensor_1 = Tensor[dtype](1)
        tensor_1[0] = 1
        let n_1 = Node[dtype](tensor_1)
        let n_targets = Node[dtype](targets)

        let logout = LOG.forward(outputs)        
        let targets_logout = MUL.forward(n_targets, logout)
        let logout_1min = LOG.forward(SUB.forward(n_1, outputs))
        let targets_1min = SUB.forward(n_1, n_targets)
        let targets_logout_1min = MUL.forward(targets_1min, logout_1min)
        let entropy = SUM.forward(ADD.forward(targets_logout, targets_logout_1min))
        let negDivN: SIMD[dtype, 1] = (-1/outputs.tensor.num_elements()).cast[dtype]()
        return MUL.forward(entropy, negDivN)
    
    fn __call__(inout self, inout outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        return self.forward(outputs, targets)


# <------------SOFTMAXCROSSENTROPY------------>
# TODO
