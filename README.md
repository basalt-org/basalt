<h1 align='center'><b>Dainemo</b></h1>
<p align='center'>
    A Machine Learning framework from scratch in Mojo 🔥
</p>

<p align="center">
  <img src="dainemo.png" alt="Dainemo Logo" width="300"/>
</p>


### Examples Status

| Task       | Dataset | Forward | Backward | Training |
|------------|---------|---------|----------|----------|
| REGRESSION |   ✅    |   ✅    |    ✅    |    ✅    |
| MNIST      |   ✅    |   ❌    |    ❌    |    ❌    |


<sub>Regression example:</sub>
<p>
  <img src="./dainemo.gif" alt="Dainemo Logo" width="400"/>
</p>



# Getting started

Running a regression example

```
mojo run -I . examples/housing.mojo
```

Compare to a alternative PyTorch implementation:  
Install the requirements in `python-requirements.txt`

```
python examples/housing.py
```

# Progress

❌: Not implemented  
✅: Working (but might require changes because of not implemented dependencies)  
WIP: Work in progress  

### Autograd

| Task        | Status |
|-------------|--------|
| NODE        |   ✅   |
| GRAPH       |   ✅   |

### Operators

| Task       | Status |
|------------|--------|
| ADD        |   ✅   |
| SUB        |   ✅   |
| MUL        |   ✅   |
| DIV        |   ✅   |
| DOT        |   ✅   |
| EXP        |   ✅   |
| LOG        |   ✅   |
| POW        |   ✅   |
| SUM        |   ✅*  |
| TRANSPOSE  |   ✅   |
| FLATTEN    |   ✅   |
| RESHAPE    |   ✅   |
| CONV2D     |   ✅   |
| CONV3D     |   ❌   |
| MAXPOOL2D  |   ✅   |
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
| RELU      |   ✅   |
| SIGMOID   |   ✅   |
| TANH      |   ✅   |
| SOFTMAX   |   ❌   |
| LEAKYRELU |   ❌   |

### Optimizers

| Task  | Status |
|-------|--------|
| ADAM  |   ✅   |

### Layers

| Task       | Status |
|------------|--------|
| SEQUENTIAL |   ❌   |
| LINEAR     |   ✅   |
| DROPOUT    |   ❌   |
| CONV2D     |   ✅   |
| CONV3D     |   ❌   |
| MAXPOOL2D  |   ✅   |
| MAXPOOL3D  |   ❌   |

### Other

| Task                          | Status |
|-------------------------------|--------|
| Model abstractions (eval/save/load/...) |   ❌   |
| Datasets (MNIST/Boston Housing)         |   ✅   |
| Dataloader                    |   ✅   |
| Tensorutils                   |   ✅   |
| Checkpoints                   |   ❌   |

### Datasets

| Task       | Original | Included Kaggle CSV |
|------------|----------|-----------------------|
| **Bosten Housing Dataset**  |   [names](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names), [data](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data) | [kaggle](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset) |
| **MNIST DataSet**     |   [training images](https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz), [training labels](https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz), <br>[test images](https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz), [test labels](https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)    | (subset of) [kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) |


## Contributing

**Policy WIP.**

This project is in active development, and we welcome contributions:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear, descriptive commit messages.
4. Push your branch to GitHub and submit a pull request.

We appreciate any and all contributions, whether they're for bug fixes, new features, or documentation improvements.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

A huge thanks to [pranftw](https://github.com/pranftw) for the inspiration and support! This project is inspired by [neograd](https://github.com/pranftw/neograd).