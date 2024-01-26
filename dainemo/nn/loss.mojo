from tensor import Tensor
from math import add
from math.limit import max_finite

from dainemo.autograd.node import Node
from dainemo.autograd.ops.basics import ADD, SUM, SUB, DIV, EXP, MAX, LOG, POW, MUL
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
        # -1/N * sum( targets * log_softmax(outputs) )

        # LogSoftmax
        let max_values = MAX.forward[axis=1](outputs)
        let input_minus_max = SUB.forward(outputs, max_values)
        let exp_values = EXP.forward(input_minus_max)
        let sum_values = SUM.forward[axis=1](exp_values)
        let log_values = LOG.forward(sum_values)
        let log_softmax = SUB.forward(input_minus_max, log_values)
        
        # CrossEntropy (reduction Mean)
        let targets_log_softmax = MUL.forward(log_softmax, Node[dtype](targets))
        let ret = SUM.forward(targets_log_softmax)
        let negDivN: SIMD[dtype, 1] = (-1/outputs.tensor.num_elements()).cast[dtype]()
        return MUL.forward(ret, negDivN)

    fn __call__(inout self, inout outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
        return self.forward(outputs, targets)
