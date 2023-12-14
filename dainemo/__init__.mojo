from .autograd.graph import Graph

alias dtype = DType.float32
var GRAPH = Graph[dtype, tracking=True]()