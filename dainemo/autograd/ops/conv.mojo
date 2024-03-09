from tensor import TensorShape

# from dainemo import GRAPH
# from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import calculate_strides
from dainemo.autograd.attributes import Attribute, AttributeVector


# <------------GENERAL CONV METHODS------------>
@always_inline
fn get_result_shape(
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        padding: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2]
    ) -> StaticIntTuple[2]:
    """
    Calculates the X and Y dimensions of the resulting convolution.
    Dimensions X, Y are on the end of the shape (..., X, Y)
        dimension X on index -2.
        dimension Y on index -1.
    """

    var result_x_dim = (
        (input_shape[-2] + (2 * padding[0]) - dilation[0] * (kernel_shape[-2] - 1) - 1)
        // stride[0]
    ) + 1
    var result_y_dim = (
        (input_shape[-1] + (2 * padding[1]) - dilation[1] * (kernel_shape[-1] - 1) - 1)
        // stride[1]
    ) + 1

    return StaticIntTuple[2](result_x_dim, result_y_dim)


# <------------CONV2D------------>
struct CONV2D:
    @staticmethod
    fn result_shape(input_shape: TensorShape, kernel_shape: TensorShape, bias_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        # Output shape = [batch, out_channels, oX, oY]
        
        var padding = attributes["padding"].value().to_static[2]()
        var stride = attributes["stride"].value().to_static[2]()
        var dilation = attributes["dilation"].value().to_static[2]()
        var res = get_result_shape(input_shape, kernel_shape, padding, stride, dilation)

        return TensorShape(input_shape[0], kernel_shape[0], res[0], res[1])

    @staticmethod
    fn forward[
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector
    ](inout outputs: Tensor[dtype], inputs: Tensor[dtype], kernel: Tensor[dtype], bias: Tensor[dtype]):
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
            inputs.shape     [batch, in_channels, iX, iY]
            kernel.shape     [out_channels, in_channels, kX, kY] (or weights)
            bias.shape       [out_channels].
            output.shape     [batch, out_channels, oX, oY].
        """
        alias padding = attributes["padding"].value().to_static[2]()
        alias stride = attributes["stride"].value().to_static[2]()
        alias dilation = attributes["dilation"].value().to_static[2]()
        
        alias inputs_strides = calculate_strides(input_shape)
        alias kernel_strides = calculate_strides(kernel_shape)
        alias output_shape = Self.result_shape(input_shape, kernel_shape, bias_shape, attributes)
        alias outputs_strides = calculate_strides(output_shape)

        @parameter
        fn kernel_iteration[
            all_checks: Bool = True
        ](batch: Int, out_ch: Int, x: Int, y: Int):
            var result: SIMD[dtype, 1] = 0

            var ix_base = x * stride[0] - padding[0]
            var iy_base = y * stride[1] - padding[1]
            for in_ch in range(input_shape[1]):
                for kx in range(kernel_shape[2]):
                    for ky in range(kernel_shape[3]):
                        var ix = ix_base + kx * dilation[0]
                        var iy = iy_base + ky * dilation[1]

                        @parameter
                        if all_checks:
                            if (
                                ix < 0
                                or iy < 0
                                or ix >= input_shape[2]
                                or iy >= input_shape[3]
                            ):
                                continue

                        var kernel_index = (
                            out_ch * kernel_strides[0]
                            + in_ch * kernel_strides[1]
                            + kx * kernel_strides[2]
                            + ky
                        )

                        var input_index = (
                            batch * inputs_strides[0]
                            + in_ch * inputs_strides[1]
                            + ix * inputs_strides[2]
                            + iy
                        )

                        result += (
                            inputs[input_index] * kernel[kernel_index]
                        )

            var output_index = (
                batch * outputs_strides[0]
                + out_ch * outputs_strides[1]
                + x * outputs_strides[2]
                + y
            )

            outputs[output_index] = result + bias[out_ch]

        alias oH_border_0 = 0
        alias oH_border_1 = (padding[0] + stride[0] + 1) // stride[0]
        alias oH_border_2 = (
            input_shape[2] + padding[0] - kernel_shape[2] * dilation[0]
        ) // stride[0]
        alias oH_border_3 = output_shape[2]

        alias oW_border_0 = 0
        alias oW_border_1 = (padding[1] + stride[0] + 1) // stride[1]
        alias oW_border_2 = (
            input_shape[3] + padding[1] - kernel_shape[3] * dilation[1]
        ) // stride[1]
        alias oW_border_3 = output_shape[3]

        for batch in range(input_shape[0]):
            for out_ch in range(output_shape[1]):
                var batch_o_idx = batch * outputs_strides[0]
                var out_ch_o_idx = out_ch * outputs_strides[1]
                # Case 1: oh might put us out of bounds
                for x in range(oH_border_0, oH_border_1):
                    for y in range(output_shape[3]):
                        kernel_iteration(batch, out_ch, x, y)
                # Case 2: oh in bounds
                for x in range(oH_border_1, oH_border_2):
                    # Case a: ow might put us out of bounds
                    for y in range(oW_border_0, oW_border_1):
                        kernel_iteration(batch, out_ch, x, y)
                    # Case b: ow in bounds
                    for y in range(oW_border_1, oW_border_2):
                        kernel_iteration[False](batch, out_ch, x, y)
                    # Case c: ow might put us out of bounds
                    for y in range(oW_border_2, oW_border_3):
                        kernel_iteration(batch, out_ch, x, y)
                # Case 3: oh might put us out of bounds
                for x in range(oH_border_2, oH_border_3):
                    for y in range(output_shape[3]):
                        kernel_iteration(batch, out_ch, x, y)

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector
    ](ug: Tensor[dtype], inputs: Tensor[dtype], kernel: Tensor[dtype], bias: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of 2D convolution.

        Upper gradient of shape: [batch, out_channels, uX, uY].
        """
        alias padding = attributes["padding"].value().to_static[2]()
        alias stride = attributes["stride"].value().to_static[2]()
        alias dilation = attributes["dilation"].value().to_static[2]()

        alias inputs_strides = calculate_strides(input_shape)
        alias kernel_strides = calculate_strides(kernel_shape)
        alias ug_strides = calculate_strides(ug_shape)

        var res: Tensor[dtype]

        @parameter
        if tensor_id == 0:
            # Inputs
            res = Tensor[dtype](input_shape)

            for batch in range(input_shape[0]):
                for out_ch in range(ug_shape[1]):
                    for ux in range(ug_shape[2]):
                        for uy in range(ug_shape[3]):
                            var ix_base = ux * stride[0] - padding[0]
                            var iy_base = uy * stride[1] - padding[1]
                            for in_ch in range(input_shape[1]):
                                for kx in range(kernel_shape[2]):
                                    for ky in range(kernel_shape[3]):
                                        var ix = ix_base + kx * dilation[0]
                                        var iy = iy_base + ky * dilation[1]

                                        if (
                                            ix < 0
                                            or iy < 0
                                            or ix >= input_shape[2]
                                            or iy >= input_shape[3]
                                        ):
                                            continue

                                        var kernel_index = (
                                            out_ch * kernel_strides[0]
                                            + in_ch * kernel_strides[1]
                                            + kx * kernel_strides[2]
                                            + ky
                                        )

                                        var ug_index = (
                                            batch * ug_strides[0]
                                            + out_ch * ug_strides[1]
                                            + ux * ug_strides[2]
                                            + uy
                                        )

                                        var input_index = (
                                            batch * inputs_strides[0]
                                            + in_ch * inputs_strides[1]
                                            + ix * inputs_strides[2]
                                            + iy
                                        )
                                        res[input_index] += (
                                            kernel[kernel_index] * ug[ug_index]
                                        )

        elif tensor_id == 1:
            # Kernel
            res = Tensor[dtype](kernel_shape)

            for in_ch in range(input_shape[1]):
                for out_ch in range(ug_shape[1]):
                    for kx in range(kernel_shape[2]):
                        for ky in range(kernel_shape[3]):
                            var result: SIMD[dtype, 1] = 0
                            for batch in range(input_shape[0]):
                                for ux in range(ug_shape[2]):
                                    for uy in range(ug_shape[3]):
                                        var ix = ux * stride[0] - padding[0] + kx * dilation[0]
                                        var iy = uy * stride[1] - padding[1] + ky * dilation[1]

                                        if (
                                            ix < 0
                                            or iy < 0
                                            or ix >= input_shape[2]
                                            or iy >= input_shape[3]
                                        ):
                                            continue

                                        var input_index = (
                                            batch * inputs_strides[0]
                                            + in_ch * inputs_strides[1]
                                            + ix * inputs_strides[2]
                                            + iy
                                        )

                                        var ug_index = (
                                            batch * ug_strides[0]
                                            + out_ch * ug_strides[1]
                                            + ux * ug_strides[2]
                                            + uy
                                        )

                                        result += inputs[input_index] * ug[ug_index]

                            var kernel_index = (
                                out_ch * kernel_strides[0]
                                + in_ch * kernel_strides[1]
                                + kx * kernel_strides[2]
                                + ky
                            )
                            res[kernel_index] = result

        else:
            # Bias
            # Sum of upper gradient over batch and X, Y dimensions
            # out_channels == ug_shape[1] == bias_shape[0]
            res = Tensor[dtype](bias_shape)

            for out_ch in range(ug_shape[1]):
                var sum: SIMD[dtype, 1] = 0
                for batch in range(ug_shape[0]):
                    for ux in range(ug_shape[2]):
                        for uy in range(ug_shape[3]):
                            var ug_index = (
                                batch * ug_strides[0]
                                + out_ch * ug_strides[1]
                                + ux * ug_strides[2]
                                + uy
                            )
                            sum += ug[ug_index]

                res[out_ch] = sum

        return res


# # <------------CONV3D------------>
# # TODO