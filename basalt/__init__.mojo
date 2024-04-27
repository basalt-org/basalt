from .autograd import Graph, Symbol, OP
from .nn import Tensor, TensorShape
from basalt.utils.collection import Collection

alias dtype = DType.float32
alias nelts = 2 * simdwidthof[dtype]()
alias seed = 42
