from .autograd import Graph, Symbol, OP


alias dtype = DType.float32
alias nelts = 2*simdwidthof[dtype]()
alias seed = 42

# 8 is the maximum rank Symbol can handle
# required as the register should have a static size while this still
# needs to work for tensors of different ranks
alias max_rank = 8