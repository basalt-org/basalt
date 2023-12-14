from random import rand
from tensor import Tensor
from testing import assert_equal

import dainemo.nn as nn
from dainemo import GRAPH
from dainemo.autograd.node import Node

alias dtype = DType.float32


# <------------LINEAR------------>
fn test_linear() raises:
    var f = nn.Linear(5, 3)                             # 5 inputs, 3 outputs
    
    let inputs: Tensor[dtype] = rand[dtype](2, 5)       # A batch of 2 with 5 inputs
    
    let outputs = f(Node[dtype](inputs))                # Should be a batch of 2 with 3 outputs

    print("Input batch of 2 with 5 inputs:", inputs.shape())
    print("Output batch of 2 with 3 outputs:", outputs.tensor.shape())
    assert_equal(outputs.tensor.dim(0), 2)
    assert_equal(outputs.tensor.dim(1), 3)
    
    # 5 Nodes added to the graph
    # inputs, weights, output(inputsxweight), bias, output(output+bias)
    assert_equal(GRAPH.graph.size, 5)


fn main():

    try:
        test_linear()
    except:
        print("[ERROR] Error in layers")
