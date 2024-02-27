# from utils.index import Index
# from tensor import Tensor, TensorShape

# import dainemo.nn as nn
# from dainemo.autograd.node import Node
# from dainemo.utils.datasets import MNIST
# from dainemo.utils.dataloader import DataLoader
# from dainemo.autograd.ops.basics import RESHAPE

# alias dtype = DType.float32



# def plot_image[dtype: DType](borrowed data: Tensor[dtype], num: Int):
#     from python.python import Python, PythonObject
    
#     np = Python.import_module("numpy")
#     plt = Python.import_module("matplotlib.pyplot")

#     let pyimage: PythonObject = np.empty((28, 28), np.float64)
#     for m in range(28):
#         for n in range(28):
#             pyimage.itemset((m, n), data[Index(num, 0, m, n)])

#     plt.imshow(pyimage)
#     plt.show()



# struct CNN:
#     var l1: nn.Conv2d[2, 1]
#     var l2: nn.ReLU
#     var l3: nn.MaxPool2d[2]
#     var l4: nn.Conv2d[2, 1]
#     var l5: nn.ReLU
#     var l6: nn.MaxPool2d[2]
#     var l7: nn.Linear

#     fn __init__(inout self):
#         self.l1 = nn.Conv2d[2, 1](
#             in_channels=1,
#             out_channels=16,
#             kernel_size=5
#         )
#         self.l2 = nn.ReLU()
#         self.l3 = nn.MaxPool2d[kernel_size=2]()
#         self.l4 = nn.Conv2d[2, 1](
#             in_channels=16, 
#             out_channels=32,
#             kernel_size=5
#         )
#         self.l5 = nn.ReLU()
#         self.l6 = nn.MaxPool2d[kernel_size=2]()
#         self.l7 = nn.Linear(n_input = 32*7*7, n_output=10)
        
#     fn forward(inout self, x: Tensor[dtype]) -> Node[dtype]:
#         var output = self.l1(Node[dtype](x))
#         output = self.l2(output)
#         output = self.l3(output)
#         output = self.l4(output)
#         output = self.l5(output)
#         output = self.l6(output)
#         output = RESHAPE.forward(output, TensorShape(output.tensor.dim(0), 32*7*7))
#         output = self.l7(output)
#         return output


# fn main():    
#     alias num_epochs = 20
#     alias batch_size = 4
#     alias learning_rate = 1e-3
    
    
#     let train_data: MNIST[dtype]
#     try:
#         train_data = MNIST[dtype](file_path='./examples/data/mnist_test_small.csv')
#         # _ = plot_image[dtype](train_data.data, 1)
#     except:
#         print("Could not load data")

#     var training_loader = DataLoader[dtype](
#                             data=train_data.data,
#                             labels=train_data.labels,
#                             batch_size=batch_size
#                         )


#     var model = CNN()
#     var loss_func = nn.CrossEntropyLoss()
#     var optim = nn.optim.Adam(lr=learning_rate)

#     let batch_data: Tensor[dtype]
#     let batch_labels: Tensor[dtype]
#     for epoch in range(num_epochs):
#         var num_batches: Int = 0
#         var epoch_loss: Float32 = 0.0
#         for batch in training_loader:
                        
#             # try:
#             #     _ = plot_image[dtype](batch.data, 0)
#             # except: 
#             #     print("Could not plot image")

#             # Forward pass
#             var output = model.forward(batch.data)
            
#             # [ONE HOT ENCODING!]
#             var labels_one_hot = Tensor[dtype](batch.labels.dim(0), 10)
#             for bb in range(batch.labels.dim(0)):
#                 labels_one_hot[Index(bb, batch.labels[bb])] = 1.0
            
#             var loss = loss_func(output, labels_one_hot)

#             # Backward pass
#             optim.zero_grad()
#             loss.backward()
#             optim.step()

#             epoch_loss += loss.tensor[0]
#             num_batches += 1

#             print("Epoch [", epoch + 1, "/", num_epochs, "] \t Step [", num_batches, "/", training_loader._num_batches, "] \t Loss: ", epoch_loss / num_batches)
