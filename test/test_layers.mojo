from random import rand
from tensor import Tensor


import dainemo.nn as nn
from dainemo.autograd.graph import Graph
from dainemo.autograd.node import Node


fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()
    var graph = Graph[dtype]()

    # <------------LINEAR------------>
    var f = nn.Linear[dtype](5, 3)
    
    let inputs: Tensor[dtype] = rand[dtype](2, 5)    # A batch of 2 with 5 inputs
    
    let outputs = f(graph, Node[dtype](inputs))

    print("Lindear layer with 5 inputs and 3 outputs.")
    print("Input batch of 2 with 5 inputs:", inputs.shape().__str__())
    print("Output batch of 2 with 3 outputs:", outputs.tensor.shape().__str__())

    print("5 Nodes added to the graph: ", graph.graph.size)   # inputs, weights, output(inputsxweight), bias, output(output+bias)
    for graph_node in graph.graph:
        print("Children:", graph_node.children.size, "Parents:", graph_node.parents.size)
        