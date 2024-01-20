from tensor import Tensor

from dainemo.autograd.node import Node
from dainemo.autograd.ops.basics import SUM, SUB, DIV, EXP, MAX

'''Activation functions.'''

# <------------RELU------------>
struct RELU:
    pass


# <------------SIGMOID------------>



# <------------TANH------------>



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


# <------------LEAKYRELU------------>