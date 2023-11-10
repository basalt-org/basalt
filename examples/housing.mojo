from dainemo.utils.datasets import BostonHousing
from dainemo.utils.dataloader import DataLoader

import dainemo.nn as nn
from dainemo.autograd.graph import Graph
from dainemo.autograd.node import Node

from tensor import Tensor, TensorShape
from utils.index import Index



struct LinearRegression[dtype: DType]:
    var graph: Graph[dtype]
    var layer1: nn.Linear[dtype]

    fn __init__(inout self, input_dim: Int):
        self.graph = Graph[dtype]()
        self.layer1 = nn.Linear[dtype](input_dim, 1)
        
    fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
        return self.layer1(self.graph, Node[dtype](x))



fn main():
    alias dtype = DType.float32
    
    let train_data: BostonHousing[dtype]
    try:
        train_data = BostonHousing[dtype](file_path='./examples/data/housing.csv')
    except:
        print("Could not load data")

    # print(train_data.data)
    
    alias num_epochs = 1
    alias batch_size = 4
    var training_loader = DataLoader[dtype](
                            data=train_data.data,
                            labels=train_data.labels,
                            batch_size=batch_size
                        )
    
    
    var model = LinearRegression[dtype](train_data.data.dim(1))
    var loss_func = nn.MSELoss[dtype]()


    let batch_start: Int
    let batch_end: Int
    let batch_data: Tensor[dtype]
    let batch_labels: Tensor[dtype]
    for epoch in range(num_epochs):
        for batch_indeces in training_loader:
            
            batch_start = batch_indeces.get[0, Int]()
            batch_end = batch_indeces.get[1, Int]()

            batch_data = create_data_batch[dtype](batch_start, batch_end, training_loader.data)
            batch_labels = create_label_batch[dtype](batch_start, batch_end, training_loader.labels)
            
            let output = model.forward(batch_data)            
            let loss = loss_func(model.graph, output, batch_labels)

            print("output")
            print(output.tensor)
            
            print("loss")
            print(loss.tensor)

            print("#####  GRAPH  #####", model.graph.graph.size)

            for graph_node in model.graph.graph:
                print("\n ----------------")
                print(graph_node.node.tensor)
                print(graph_node.backward_fn)
                print(graph_node.node.requires_grad)
                print(graph_node.node.uuid)
                print('\t Parents,')
                for parent in graph_node.parents:
                    print('\t\t', parent.uuid)
                print('\t Children,')
                for child in graph_node.children:
                    print('\t\t', child.uuid)

            break
        break



#TODO: See DataLoader
fn create_data_batch[dtype: DType](start: Int, end: Int, data: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start, 13))
    for n in range(end - start):
        for i in range(13):
            batch[Index(n, i)] = data[Index(start + n, i)]
    return batch

fn create_label_batch[dtype: DType](start: Int, end: Int, labels: Tensor[dtype]) ->  Tensor[dtype]:
    var batch = Tensor[dtype](TensorShape(end - start, 1))    
    for i in range(end - start):
        batch[i] = labels[start + i]
    return batch
