from tensor import Tensor, TensorShape
from math import floor

from dainemo import GRAPH
from dainemo.autograd.node import Node


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


fn pad(data: Tensor[dtype]):
    """
    Pads the input tensor in x and y dimensions (last two dims).
    Only the last two dimensions are padded rest aren't padded as
    (padded with 0, has no effect).
    """
    # TODO
    pass


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
    fn forward[padding: Int, stride: Int](inputs: Tensor[dtype], kernel: Tensor[dtype], bias: Tensor[dtype]) -> Node[dtype]:
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
        """
        
        # inputs.shape should be len 3 with [number of examples, X, Y]

        let result_shape = get_result_shape[padding, stride](inputs.shape(), kernel.shape())

        let outputs = Tensor[dtype](inputs.dim(0), result_shape[0], result_shape[1])

        print(outputs)


# <------------CONV3D------------>


# <------------MAXPOOL2D------------>


# <------------MAXPOOL3D------------>