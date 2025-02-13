from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.autograd.attributes import Attribute, AttributeVector


# '''Activation functions.'''
fn ReLU(inout g: Graph, input: Symbol) -> Symbol:
    return g.op(OP.RELU, input)


fn LeakyReLU(
    inout g: Graph, input: Symbol, negative_slope: Scalar[dtype]
) -> Symbol:
    return g.op(
        OP.LEAKYRELU,
        input,
        attributes=AttributeVector(Attribute("negative_slope", negative_slope)),
    )

fn GELU(inout g: Graph, input: Symbol) -> Symbol:
    var SQRT_2_OVER_PI = 0.7978845608028654
    var GELU_COEFF = 0.044715

    var x_cubed = g.op(OP.POW, input, 3.0)
    var term = g.op(OP.ADD, input, g.op(OP.MUL, GELU_COEFF, x_cubed))
    var scaled_term = g.op(OP.MUL, SQRT_2_OVER_PI, term)
    var tanh_result = g.op(OP.TANH, scaled_term)
    var one_plus_tanh = g.op(OP.ADD, 1.0, tanh_result)
    var gelu_output = g.op(OP.MUL, g.op(OP.MUL, 0.5, input), one_plus_tanh)

    return gelu_output



fn Sigmoid(inout g: Graph, input: Symbol) -> Symbol:
    return g.op(OP.SIGMOID, input)


fn Tanh(inout g: Graph, input: Symbol) -> Symbol:
    return g.op(OP.TANH, input)


fn Softmax(inout g: Graph, input: Symbol, axis: Int) -> Symbol:
    # softmax: exp(x_i) / sum(exp(x_j))
    # stable softmax: exp(x_i - max(x_j)) / sum(exp(x_j - max(x_j)))

    var max_values = g.op(
        OP.MAX, input, attributes=AttributeVector(Attribute("axis", axis))
    )
    var input_minus_max = g.op(OP.SUB, input, max_values)
    var exp_values = g.op(OP.EXP, input_minus_max)
    var sum_values = g.op(
        OP.SUM, exp_values, attributes=AttributeVector(Attribute("axis", axis))
    )

    return g.op(OP.DIV, exp_values, sum_values)


fn LogSoftmax(inout g: Graph, input: Symbol, axis: Int) -> Symbol:
    # stable logsoftmax: log(exp(x_i - max(x_j)) / sum(exp(x_j - max(x_j))))
    # stable logsoftmax: x_i - max(x_j) - log(sum(exp(x_j - max(x_j))))

    var max_values = g.op(
        OP.MAX, input, attributes=AttributeVector(Attribute("axis", axis))
    )
    var input_minus_max = g.op(OP.SUB, input, max_values)
    var exp_values = g.op(OP.EXP, input_minus_max)
    var sum_values = g.op(
        OP.SUM, exp_values, attributes=AttributeVector(Attribute("axis", axis))
    )
    var log_values = g.op(OP.LOG, sum_values)

    return g.op(OP.SUB, input_minus_max, log_values)
