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
    ](
        inputs: Tensor[dtype], kernel: Tensor[dtype], bias: Tensor[dtype]
    ) -> Node[dtype]:
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
            inputs.shape     [batch, in_channels, X, Y]
            kernel.shape     [out_channels, in_channels, X, Y]
            bias.shape       [out_channels].
        """
        # TODO: Add bias

        alias nelts: Int = simdwidthof[dtype]()

        let result_shape = get_result_shape[padding, stride](
            inputs.shape(), kernel.shape()
        )

        var outputs = Tensor[dtype](inputs.dim(0), kernel.dim(0), result_shape[0], result_shape[1])

        for batch in range(inputs.dim(0)):
            for out_ch in range(kernel.dim(0)):
                # (** split) OPTIMIZATION
                for x in range(outputs.dim(2)):
                    for y in range(outputs.dim(3)):
                        var result: SIMD[dtype, 1] = 0
                        for in_ch in range(inputs.dim(1)):
                            for kx in range(kernel.dim(2)):
                                for ky in range(kernel.dim(3)):
                                    let ix = x * stride - padding + kx
                                    let iy = y * stride - padding + ky

                                    ### TODO: OPTIMIZATION
                                    # Split into edge cases to avoid having to check bounds, by determening the borders (todo)
                                    # Case 1: TOP       x might put us out of bounds (check)
                                    # Case 2: LEFT      y might put us out of bounds (check)
                                    # Case 3: Base case, in bounds (no checking needed, in the majority of cases)
                                    # Case 4: RIGHT     y might put us out of bounds (check)
                                    # Case 5: BOTTOM    x might put us out of bounds (check)
                                    
                                    if not (
                                        ix < 0 or iy < 0 
                                        or ix >= inputs.dim(2) 
                                        or iy >= inputs.dim(3)
                                    ):

                                        let input_index = (
                                            batch * (inputs.dim(1) * inputs.dim(2) * inputs.dim(3)) 
                                            + in_ch * (inputs.dim(2) * inputs.dim(3)) 
                                            + ix * inputs.dim(2) 
                                            + iy
                                        )
                                        let kernel_index = (
                                            out_ch * (inputs.dim(1) * kernel.dim(2) * kernel.dim(3)) 
                                            + in_ch * (kernel.dim(2) * kernel.dim(3)) 
                                            + kx * kernel.dim(2) 
                                            + ky
                                        )
                                        result += inputs[input_index] * kernel[kernel_index]

                        let output_index = (
                            batch * (kernel.dim(0) * outputs.dim(2) * outputs.dim(3)) 
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