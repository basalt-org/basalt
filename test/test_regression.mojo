from random import rand

import dainemo.nn as nn
from dainemo.autograd.graph import Graph
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import fill


struct LinearRegression[dtype: DType]:
    var graph: Graph[dtype]
    var layer1: nn.Linear[dtype]

    fn __init__(inout self, input_dim: Int):
        self.graph = Graph[dtype]()
        self.layer1 = nn.Linear[dtype](self.graph, input_dim, 1)
        
    fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
        return self.layer1(self.graph, Node[dtype](x))



fn replace_param[dtype: DType](inout g: Graph[dtype], param_idx: Int, cst_value: FloatLiteral):
    alias nelts: Int = simdwidthof[dtype]()
    var param = g.parameters.get(param_idx)
    var new_param_tensor = Tensor[dtype](param.tensor.shape())
    fill[dtype, nelts](new_param_tensor, cst_value)
    param.tensor = new_param_tensor
    g.parameters.replace(param_idx, param)



fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()

    let batch_size = 2
    let input_size = 4

    var model = LinearRegression[dtype](input_size)
    var loss_func = nn.MSELoss[dtype]()
    var optim = nn.optim.Adam[dtype](lr=0.05)

    # Overwrite parameters of the model
    replace_param(model.graph, 0, 1.0)
    replace_param(model.graph, 1, 1.0)
    print(model.graph.parameters.get(0).tensor)
    print(model.graph.parameters.get(1).tensor)

    # Create batch data
    var batch_data = Tensor[dtype](batch_size, input_size)
    var batch_labels = Tensor[dtype](batch_size, 1)
    fill[dtype, nelts](batch_data, 1.0)
    fill[dtype, nelts](batch_labels, 1.0)
    print(batch_data)
    print(batch_labels)

    #### FORWARD ####
    optim.zero_grad(model.graph)
    let output = model.forward(batch_data)
    var loss = loss_func(model.graph, output, batch_labels)
    print("OUTPUT: ", output.tensor)
    print("LOSS: ", loss.tensor[0])

    #### BACKWARD ####

    print("#####  GRADIENTS Before  #####", model.graph.parameters.size)
    for param in model.graph.parameters:
        print("\n ----------------")
        print("\t Grad:", param.grad)

    print("------------ BACKWARD ------------")
    loss.backward(model.graph)
    

    print("#####  GRADIENTS AFTER  #####", model.graph.parameters.size)
    for param in model.graph.parameters:
        print("Grad:", param.grad)

    optim.step(model.graph)

    print("#####  PARAMETERS  #####", model.graph.parameters.size)
    for param in model.graph.parameters:
        print("Param:", param.tensor)
