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
        padding: Int, stride: Int
    ](
        inputs: Tensor[dtype], kernel: Tensor[dtype], bias: Tensor[dtype]
    ) -> Node[dtype]:
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
        """
        # NOTE: (for now) inputs.shape should be len 3 with [batch, X, Y]
        # TODO 1: Add bias
        # TODO 2: Support in_channels and out_channels
        #         inputs.shape should be len 4 with [batch, in_channels, X, Y]
        #         kernel.shape should be len 4 with [out_channels, in_channels, X, Y]
        #         bias.shape should be len 1 with [out_channels]

        alias nelts: Int = simdwidthof[dtype]()

        let result_shape = get_result_shape[padding, stride](
            inputs.shape(), kernel.shape()
        )
        var outputs = Tensor[dtype](inputs.dim(0), result_shape[0], result_shape[1])
        let padded_inputs = pad[padding](inputs)

        var index_i = 0
        var index_j = 0
        let kernel_x_dim = kernel.shape()[-2]
        let kernel_y_dim = kernel.shape()[-1]

        for batch in range(outputs.dim(0)):
            for i in range(outputs.dim(1)):
                for j in range(outputs.dim(2)):
                    # Iterate over kernel and multiply with fragment
                    var result: SIMD[dtype, 1] = 0
                    for k in range(kernel_x_dim):
                        for l in range(kernel_y_dim):
                            let padded_index = (
                                batch * padded_inputs.dim(1) * padded_inputs.dim(2)
                                + (index_i + k) * padded_inputs.dim(2)
                                + (index_j + l)
                            )
                            let kernel_index = (k * kernel.dim(1) + l)
                            result += padded_inputs[padded_index] * kernel[kernel_index]

                    let output_index = (
                        batch * outputs.dim(1) * outputs.dim(2) + i * outputs.dim(2) + j
                    )
                    outputs[output_index] = result

                    # Increment index by stride
                    index_j += stride
                index_j = 0
                index_i += stride
            index_i = 0

        return GRAPH.create_graph_node[Self.backward](outputs, inputs, kernel, bias)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of 2D convolution."""
        # TODO
        return Tensor[dtype]()



# <------------CONV3D------------>


# <------------MAXPOOL2D------------>


# <------------MAXPOOL3D------------>