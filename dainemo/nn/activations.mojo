from tensor import Tensor

from dainemo.autograd.node import Node
from dainemo.autograd.ops.basics import SUM, SUB, DIV, EXP, MAX, LOG
import dainemo.autograd.ops.mlops

'''Activation functions.'''

# <------------RELU------------>
struct RELU:
    @staticmethod
    fn forward(input: Node[dtype]) -> Node[dtype]:
        return mlops.RELU.forward(input)


# <------------SIGMOID------------>
struct Sigmoid:
    @staticmethod
    fn forward(input: Node[dtype]) -> Node[dtype]:
        return mlops.SIGMOID.forward(input)


# <------------TANH------------>
struct Tanh:
    @staticmethod
    fn forward(input: Node[dtype]) -> Node[dtype]:
        return mlops.TANH.forward(input)


# <------------SOFTMAX------------>
struct Softmax:
    @staticmethod
    fn forward[axis: Int](input: Node[dtype]) -> Node[dtype]:
        # softmax: exp(x_i) / sum(exp(x_j))
        # stable softmax: exp(x_i - max(x_j)) / sum(exp(x_j))

        let max_values = MAX.forward[axis](input)
        let input_minus_max = SUB.forward(input, max_values)
        let exp_values = EXP.forward(input_minus_max)
        let sum_values = SUM.forward[axis](exp_values)

        return DIV.forward(exp_values, sum_values)


# <------------LOGSOFTMAX------------>
struct LogSoftmax:
    @staticmethod
    fn forward[axis: Int](input: Node[dtype]) -> Node[dtype]:
        # logsoftmax: log(exp(x_i) / sum(exp(x_j)))
        # logsoftmax: x_i - log(sum(exp(x_j)))

        let max_values = MAX.forward[axis](input)
        let input_minus_max = SUB.forward(input, max_values)
        let exp_values = EXP.forward(input_minus_max)
        let sum_values = SUM.forward[axis](exp_values)
        let log_values = LOG.forward(sum_values)

        return SUB.forward(input_minus_max, log_values)


# <------------LEAKYRELU------------>