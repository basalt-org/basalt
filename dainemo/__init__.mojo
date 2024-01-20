from tensor import TensorShape
from .autograd.graph import Graph

alias dtype = DType.float32
alias nelts = simdwidthof[dtype]()
alias NONE_BC = TensorShape(-1, -1)

var GRAPH = Graph[dtype, tracking=True]()