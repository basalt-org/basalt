from .autograd import Graph, Symbol, OP
from .nn import Tensor, TensorShape

alias dtype = DType.float32
alias nelts = 2 * simdwidthof[dtype]()
alias seed = 42
