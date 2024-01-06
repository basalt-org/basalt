from tensor import Tensor, TensorShape
from math import floor

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import calculate_strides


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



# <------------CONV2D------------>
struct CONV2D:
    @staticmethod
    fn forward[
        padding: Int, stride: Int
    ](inputs: Node[dtype], kernel: Node[dtype], bias: Node[dtype]) -> Node[dtype]:
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
            inputs.shape     [batch, in_channels, X, Y]
            kernel.shape     [out_channels, in_channels, X, Y] (or weights)
            bias.shape       [out_channels].
            output.shape     [batch, out_channels, X, Y].
        """
        # TODO: Add bias

        alias nelts: Int = simdwidthof[dtype]()

        let result_shape = get_result_shape[padding, stride](
            inputs.tensor.shape(), kernel.tensor.shape()
        )

        var outputs = Tensor[dtype](
            inputs.tensor.dim(0), kernel.tensor.dim(0), result_shape[0], result_shape[1]
        )

        @parameter
        fn kernel_iteration(
            batch: Int, in_ch: Int, out_ch: Int, x: Int, y: Int
        ) -> SIMD[dtype, 1]:
            var result: SIMD[dtype, 1] = 0
            for kx in range(kernel.tensor.dim(2)):
                for ky in range(kernel.tensor.dim(3)):
                    let ix = x * stride - padding + kx
                    let iy = y * stride - padding + ky

                    if ix < 0 or iy < 0 or ix >= inputs.tensor.dim(2) or iy >= inputs.tensor.dim(3):
                        continue

                    let kernel_index = (
                        out_ch * (kernel.tensor.dim(1) * kernel.tensor.dim(2) * kernel.tensor.dim(3))
                        + in_ch * (kernel.tensor.dim(2) * kernel.tensor.dim(3))
                        + kx * kernel.tensor.dim(3)
                        + ky
                    )

                    let input_index = (
                        batch * (inputs.tensor.dim(1) * inputs.tensor.dim(2) * inputs.tensor.dim(3))
                        + in_ch * (inputs.tensor.dim(2) * inputs.tensor.dim(3))
                        + ix * inputs.tensor.dim(3)
                        + iy
                    )

                    result += inputs.tensor[input_index] * kernel.tensor[kernel_index]

            return result

        for batch in range(inputs.tensor.dim(0)):
            for out_ch in range(outputs.dim(1)):

                ### TODO: OPTIMIZATION
                # Split into edge cases to avoid having to check bounds, by determening the borders (todo)
                # for (x, y) in range(border_low_<i>, border_high_<i>): ...
                # Case 1: TOP       x might put us out of bounds (check)
                # Case 2: LEFT      y might put us out of bounds (check)
                # Case 3: Base case, in bounds (no checking needed, in the majority of cases)
                # Case 4: RIGHT     y might put us out of bounds (check)
                # Case 5: BOTTOM    x might put us out of bounds (check)

                for x in range(outputs.dim(2)):
                    for y in range(outputs.dim(3)):
                        var result: SIMD[dtype, 1] = 0
                        for in_ch in range(inputs.tensor.dim(1)):
                            result += kernel_iteration(batch, in_ch, out_ch, x, y)

                        let output_index = (
                            batch * (outputs.dim(1) * outputs.dim(2) * outputs.dim(3))
                            + out_ch * (outputs.dim(2) * outputs.dim(3))
                            + x * outputs.dim(3)
                            + y
                        )
                        outputs[output_index] = result

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