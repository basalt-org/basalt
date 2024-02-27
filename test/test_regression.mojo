# from dainemo import GRAPH
# import dainemo.nn as nn
# from dainemo.autograd.node import Node
# from dainemo.utils.tensorutils import fill

# alias dtype = DType.float32
# alias nelts: Int = simdwidthof[dtype]()


# struct LinearRegression:
#     var layer1: nn.Linear

#     fn __init__(inout self, input_dim: Int):
#         self.layer1 = nn.Linear(input_dim, 1)
        
#     fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
#         return self.layer1(Node[dtype](x))



# fn main():    
#     let batch_size = 4
#     let input_size = 13

#     var model = LinearRegression(input_size)
#     var loss_func = nn.MSELoss()
#     var optim = nn.optim.Adam(lr=0.05)

#     # Create batch data
#     var batch_data = Tensor[dtype](batch_size, input_size)
#     var batch_labels = Tensor[dtype](batch_size, 1)
#     fill[dtype, nelts](batch_data, 1.0)
#     fill[dtype, nelts](batch_labels, 1.0)

#     #### FORWARD ####
#     optim.zero_grad()
#     let output = model.forward(batch_data)
#     var loss = loss_func(output, batch_labels)
#     # print("OUTPUT: ", output.tensor)
#     print("LOSS: ", loss.tensor[0])

#     #### BACKWARD ####
#     print("------------ BACKWARD ------------")
#     print("Before:", GRAPH.graph.size)
    
#     loss.backward()
    
#     print("After:", GRAPH.graph.size)
#     # for i in range(GRAPH.graph.size):
#     #     print("Gradient: ", GRAPH.graph[i].grad)

    
#     #### OPTIMIZER ####
#     print("------------ OPTIMIZER ------------")
#     print("Graph size:", GRAPH.graph.size)
#     # for i in range(GRAPH.graph.size):
#     #     print("Parameters Before: ", GRAPH.graph[i].tensor)

#     optim.step()

#     # for i in range(GRAPH.graph.size):
#     #     print("Parameters After: ", GRAPH.graph[i].tensor)


#     #### FORWARD2 ####
#     print("------------ SECOND ITER ------------")
#     optim.zero_grad()
#     let output2 = model.forward(batch_data)
#     var loss2 = loss_func(output2, batch_labels)
#     # print("OUTPUT: ", output2.tensor)
#     print("LOSS: ", loss2.tensor[0])
    
#     print("Graph size:", GRAPH.graph.size)
    
#     loss2.backward()