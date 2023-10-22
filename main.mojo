from dainemo.utils.datasets import MNIST
from dainemo.utils.dataloader import DataLoader
from dainemo.utils.tensorutils import tprint

from tensor import Tensor, TensorShape
from utils.index import Index




def plot_image[dtype: DType](borrowed data: Tensor[dtype], num: Int):
    from python.python import Python, PythonObject
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")

    let pyimage: PythonObject = np.empty((28, 28), np.float64)
    for m in range(28):
        for n in range(28):
            pyimage.itemset((m, n), data[Index(num, 0, m, n)])

    plt.imshow(pyimage)
    plt.show()




fn main():
    alias dtype = DType.float32
    
    let train_data: MNIST[dtype]
    try:
        train_data = MNIST[dtype](train=False)
        _ = plot_image[dtype](train_data.data, 1)
    except:
        print("Could not load data")


    alias num_epochs = 1
    alias batch_size = 32
    var training_loader = DataLoader[dtype](
                            data=train_data.data,
                            labels=train_data.labels,
                            batch_size=batch_size
                        )
    
    for epoch in range(num_epochs):
        var batch_count = 0
        for batch in training_loader:
            
            # TODO: construct label batch in dataloader (v0.4.0 cannot handle tuple of tensors)
            # https://github.com/modularml/mojo/issues/516
            var label_batch = Tensor[dtype](batch.dim(0))
            for i in range(batch.dim(0)):
                label_batch[i] = train_data.labels[batch_count*batch_size + i]
            batch_count += 1
            
            
            tprint[dtype](batch)
            tprint[dtype](label_batch)

            try:
                _ = plot_image[dtype](batch, 0)
            except: 
                print("Could not plot image")


            

