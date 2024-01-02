from tensor import Tensor, TensorShape
from math import floor

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import pad_zeros, tslice


# <------------GENERAL CONV METHODS------------>
fn get_result_shape[
    padding: Int,
    stride: Int
](input_shape: TensorShape, kernel_shape: TensorShape) -> StaticIntTuple[2]:
    """
    Calculates the X and Y dimensions of the resulting convolution.
    Dimensions X, Y are on the end of the shape (..., X, Y)
        dimension X on index -2.
        dimension Y on index -1.
    """

    let result_x_dim = floor[DType.float64, 1](
            ((input_shape[-2] + (2 * padding) - kernel_shape[-2]) / stride) + 1
        ).to_int()

    let result_y_dim = floor[DType.float64, 1](
            ((input_shape[-1] + (2 * padding) - kernel_shape[-1]) / stride) + 1
        ).to_int()

    return StaticIntTuple[2](result_x_dim, result_y_dim)


fn pad[padding: Int](data: Tensor[dtype]) -> Tensor[dtype]:
    """
    Pads the input tensor in x and y dimensions (last two dims).
    Only the last two dimensions are padded rest aren't padded as
    (padded with 0, has no effect).
    """
    
    # No padding for dimensions other then x and y
    var pad_width = DynamicVector[Int](data.rank() * 2)
    for _ in  range(data.rank() - 2):
        pad_width.push_back(0)
        pad_width.push_back(0)

    # Padding for x and y
    for _ in range(2):
        pad_width.push_back(padding)
        pad_width.push_back(padding)

    alias nelts: Int = simdwidthof[dtype]()
    return pad_zeros[dtype, nelts](data, pad_width)


fn unpad(padded_data: Tensor[dtype]):
    """
    Removes the padding from the padded tensor.
    Slices the padded_data in last two dimensions, 
    Rejects the padding by only extracting the original data.
    """
    # TODO
    pass


# <------------CONV2D------------>
struct CONV2D:
    @staticmethod
    fn forward[
            padding: Int,
            stride: Int
        ](inputs: Tensor[dtype], kernel: Tensor[dtype], bias: Tensor[dtype]): # -> Node[dtype]:
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
        """
        # inputs.shape should be len 3 with [batch, X, Y]
        alias nelts: Int = simdwidthof[dtype]()

        let result_shape = get_result_shape[padding, stride](inputs.shape(), kernel.shape())
        let outputs = Tensor[dtype](inputs.dim(0), result_shape[0], result_shape[1])
        let padded_inputs = pad[padding](inputs)

        var index_i = 0
        var index_j = 0
        let kernel_x_dim = kernel.shape()[-2]
        let kernel_y_dim = kernel.shape()[-1]
        for i in range(result_shape[0]):
            for j in range(result_shape[1]):

                # Get fragment of padded inputs
                let slice_x = slice(index_i, index_i + kernel_x_dim)
                let slice_y = slice(index_j, index_j + kernel_y_dim)
                # let fragment = tslice[dtype, nelts](padded_inputs, slice(0, inputs.dim(0)), slice_x, slice_y)

                # TODO
                # output = np.sum((fragment*kernel.data), axis=(1,2)) + bias.data
                # outputs[:, i, j] = output

                # Increment index by stride
                index_j += stride
            index_i += stride


# <------------CONV3D------------>


# <------------MAXPOOL2D------------>


# <------------MAXPOOL3D------------>