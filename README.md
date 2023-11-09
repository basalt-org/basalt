### Examples

| Task       | Dataset | Forward | Backward | Training |
|------------|---------|---------|----------|----------|
| REGRESSION |   ✅    |   ✅    |    ❌    |    ❌    |
| CONV       |   ✅    |   ❌    |    ❌    |    ❌    |

[SOURCES]  
**Bosten Housing Dataset**  
(original)  
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data

(Included in repo: Kaggle csv)  
https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset

**MNIST DataSet**  
(original)  
wget https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz  (training images)  
wget https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz (training labels)  
wget https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  (test images)  
wget https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  (test labels)  

(Included in repo: A subset of Kaggle csv)  
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

# Progress

❌: Not implemented  
✅: Working (but might require changes because of not implemented dependencies)  
WIP: Workin progress  

### Autograd

| Task        | Status |
|-------------|--------|
| NODE        |  WIP   |
| GRAPH       |  WIP   |

### Operators

| Task       | Status |
|------------|--------|
| ADD        |   ❌   |
| SUB        |   ❌   |
| MUL        |   ❌   |
| DIV        |   ❌   |
| DOT        |  WIP   |
| EXP        |   ❌   |
| LOG        |   ❌   |
| POW        |   ❌   |
| SUM        |   ❌   |
| TRANSPOSE  |   ❌   |
| FLATTEN    |   ❌   |
| RESHAPE    |   ❌   |
| CONV2D     |   ❌   |
| CONV3D     |   ❌   |
| MAXPOOL2D  |   ❌   |
| MAXPOOL3D  |   ❌   |

### Loss Functions

| Task      | Status |
|-----------|--------|
| MSE       |   ✅   |
| CE        |   ❌   |
| BCE       |   ❌   |
| SoftmaxCE |   ❌   |

### Activations

| Task      | Status |
|-----------|--------|
| RELU      |   ❌   |
| SIGMOID   |   ❌   |
| TANH      |   ❌   |
| SOFTMAX   |   ❌   |
| LEAKYRELU |   ❌   |

### Optimizers

| Task  | Status |
|-------|--------|
| ADAM  |   ❌   |

### Layers

| Task       | Status |
|------------|--------|
| SEQUENTIAL |   ❌   |
| LINEAR     |   ✅   |
| DROPOUT    |   ❌   |
| CONV2D     |   ❌   |
| CONV3D     |   ❌   |
| MAXPOOL2D  |   ❌   |
| MAXPOOL3D  |   ❌   |

### Other

| Task                          | Status |
|-------------------------------|--------|
| Model abstractions (eval/save/load/...) |   ❌   |
| Datasets (MNIST/Boston Housing)         |   ✅   |
| Dataloader                    |   ✅   |
| Tensorutils                   |   ✅   |
| Checkpoints                   |   ❌   |
