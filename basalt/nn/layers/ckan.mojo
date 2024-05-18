from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP
from basalt.utils import q_sqrt
from basalt.autograd.params import Param
from basalt.autograd.attributes import Attribute, AttributeVector
from basalt.nn.activations import Acos, Cos

from random import randn
from math import iota
from algorithm import vectorize


fn CKAN(
    inout g: Graph, inputs: Symbol, input_dim: Int, output_dim: Int, degree: Int
) -> Symbol:
    var coeffs = g.param(
        TensorShape(input_dim, output_dim, degree + 1),
        init=Param("random_normal", 0, 1),
    )

    var arange = g.param(
        TensorShape(degree + 1),
        init=Param("arange", 0, 0),
    )
    
    var normalized = g.op(OP.TANH, inputs)
    var reshaped = g.op(
        OP.RESHAPE,
        normalized,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(-1, input_dim, 1))
        ),
    )
    var replicated = g.concat(reshaped, degree + 1, dim=2)
    var acos = Acos(g, replicated)
    var scaled = g.op(OP.DOT, acos, arange)
    var scaled_reshaped = g.op(
        OP.RESHAPE,
        scaled,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(-1, input_dim, degree + 1))
        ),
    )
    var cos = Cos(g, scaled_reshaped)
    var unsqueezed = g.op(
        OP.UNSQUEEZE,
        cos,
        attributes=AttributeVector(Attribute("dims", TensorShape(0, 3))),
    )
    var unsqueezed_coeffs = g.op(
        OP.UNSQUEEZE,
        coeffs,
        attributes=AttributeVector(Attribute("dims", TensorShape(0))),
    )
    var mul = g.op(OP.DOT, unsqueezed, unsqueezed_coeffs)
    var summed_one = g.op(
        OP.SUM, mul, attributes=AttributeVector(Attribute("axis", 2))
    )
    var summed_two = g.op(
        OP.SUM, summed_one, attributes=AttributeVector(Attribute("axis", 1))
    )
    return g.op(
        OP.RESHAPE,
        summed_two,
        attributes=AttributeVector(
            Attribute("shape", TensorShape(-1, output_dim))
        ),
    )
