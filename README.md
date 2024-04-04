<br/>
<p align="center">
  <a href="https://github.com/Basalt-Org/Basalt">
    <img src="https://github.com/basalt-org/basalt/assets/46826967/4873806c-ff61-4903-bf3d-874d6acba3e8" alt="Logo" width="250" height="250">
  </a>

  <h1 align="center">Basalt</h1>

  <p align="center">
    A Machine Learning framework from scratch in pure Mojo 🔥
  </p>
</p>

<div align="center">
  <img src="https://img.shields.io/github/contributors/Basalt-Org/Basalt?color=dark-green" />
  <img src="https://img.shields.io/github/issues/Basalt-Org/Basalt?color=dark-green" />
  <img src="https://img.shields.io/github/license/Basalt-Org/Basalt?color=dark-green" />
</div>


## About The Project

Basalt is a stand-alone machine learning framework that leverages the power of Mojo.

As [discussed](https://docs.modular.com/mojo/why-mojo) by Modular, Mojo is a language for the future of AI development. Built on top of MLIR technology, rather than existing GCC and LLVM approaches, Mojo looks and feels like Python code, yet performs much closer to languages like Rust or C++. Parametric functions and compile time parameters allow for the graph to statically compiled. Having the static graph allows for much harder performance optimizations.

Basalt, while still in its infancy, is able to achieve speeds comparable to well established frameworks like Pytorch. Note that the project right now is mainly limited to models using Linear and Convolutional layers. Below a benchmark of the current status. But keep posted, there is much more room for improvement and we are upgrading the project on a daily basis.

![basalt_benchmark](https://github.com/basalt-org/basalt/assets/46826967/83037770-a9e3-440d-bdca-f51af0aebee0)


## Quick Start

Try out the benchmarks yourself:

```
mojo -I . examples/housing.mojo
```
```
mojo -I . examples/sin_estimate.mojo
```
```
mojo -I . examples/mnist.mojo
```

Compare to the alternative PyTorch implementation:  
Make sure to install the requirements in `python-requirements.txt` in your python environment.

```
python examples/housing.py
python examples/sin_estimate.py
python examples/mnist.py
```

## Roadmap

### v0.1.0 [x]
- [x] Improve matrix multiplication and convolution kernels
- [x] Switch to custom Tensor and TensorShape implementations
- [x] Improve benchmarks and overall model execution performance
- [x] Add profiling and additional performance tests

### v0.2.0 (WIP)
- [ ] Add additional operators: Slice, (Un)Squeeze, Concat, Clip, Gather, Split, FMA ...
- [ ] Better layer support and more activation functions
- [ ] Graph submodules & graph concatenation
- [ ] Computer vision benchmark. 

### Long-Term
- [ ] Better parallelization
- [ ] GPU support
- [ ] Reworked Dataloader
- [ ] Autotuning and related features
- [ ] Graph compilation optimizations
- [ ] Operator fusion
- [ ] ONNX / Max compatibility

## Contributing

Basalt is built by community efforts and relies on your expertise and enthousiasm!  
Small fixes and improvements are much appreciated. If you are considering larger contributions, feel free to contact us for a smoother communication channel on Discord. If you find a bug or have an idea for a feature, please use our issue tracker. Before creating a new issue, please:
* Check if the issue already exists. If an issue is already reported, you can contribute by commenting on the existing issue.
* If not, create a new issue and include all the necessary details to understand/recreate the problem or feature request.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request
> Once your changes are pushed, navigate to your fork on GitHub. And create a pull request against the original basalt-org/basalt repository.
> - Before creating a PR make sure it doesn't break any of the unit-tests. (e.g. `mojo run -I . test/test_ops.mojo`)
> - Introducing new big features require a new test!
> - In the pull request, provide a detailed description of the changes and why they're needed. Link any relevant issues.
> - If there are any specific instructions for testing or validating your changes, include those as well.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See [LICENSE](https://github.com/Basalt-Org/Basalt/blob/main/LICENSE) and the LLVM [License](https://llvm.org/LICENSE.txt) for more information.

## Acknowledgements

* Built with [Mojo](https://github.com/modularml/mojo) created by [Modular](https://github.com/modularml)
