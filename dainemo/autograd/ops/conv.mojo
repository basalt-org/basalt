from tensor import Tensor, TensorShape
from math import floor

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import calculate_strides


# <------------GENERAL CONV METHODS------------>
fn get_result_shape[
    padding: Int, stride: Int
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
        padding: Int, stride: Int, padding_mode: Int = 0
    ](inputs: Node[dtype], kernel: Node[dtype], bias: Node[dtype]) -> Node[dtype]:
        """
        Performs a 2D convolution on the input tensor using the kernel and bias.
            inputs.shape     [batch, in_channels, iX, iY]
            kernel.shape     [out_channels, in_channels, kX, kY] (or weights)
            bias.shape       [out_channels].
            output.shape     [batch, out_channels, oX, oY].
        """
        # TODO: Add bias

        alias nelts: Int = simdwidthof[dtype]()

        let result_shape = get_result_shape[padding, stride](
            inputs.tensor.shape(), kernel.tensor.shape()
        )

        var outputs = Tensor[dtype](
            inputs.tensor.dim(0), kernel.tensor.dim(0), result_shape[0], result_shape[1]
        )
        var outputs_strides = calculate_strides(outputs.shape())

        @parameter
        fn kernel_iteration_all_checks(batch: Int, out_ch: Int, x: Int, y: Int):
            var result: SIMD[dtype, 1] = 0
            for in_ch in range(inputs.tensor.dim(1)):
                for kx in range(kernel.tensor.dim(2)):
                    for ky in range(kernel.tensor.dim(3)):
                        let ix = x * stride - padding + kx
                        let iy = y * stride - padding + ky

                        let kernel_index = (
                            out_ch * kernel.strides[0]
                            + in_ch * kernel.strides[1]
                            + kx * kernel.strides[2]
                            + ky
                        )

                        if (
                            ix < 0
                            or iy < 0
                            or ix >= inputs.tensor.dim(2)
                            or iy >= inputs.tensor.dim(3)
                        ):
                            result += padding_mode * kernel.tensor[kernel_index]
                            continue

                        let input_index = (
                            batch * inputs.strides[0]
                            + in_ch * inputs.strides[1]
                            + ix * inputs.strides[2]
                            + iy
                        )

                        result += (
                            inputs.tensor[input_index] * kernel.tensor[kernel_index]
                        )

            let output_index = (
                batch * outputs_strides[0]
                + out_ch * outputs_strides[1]
                + x * outputs_strides[2]
                + y
            )

            outputs[output_index] = result + bias.tensor[out_ch]

        @parameter
        fn kernel_iteration_no_checks(batch: Int, out_ch: Int, x: Int, y: Int):
            var result: SIMD[dtype, 1] = 0
            for in_ch in range(inputs.tensor.dim(1)):
                for kx in range(kernel.tensor.dim(2)):
                    for ky in range(kernel.tensor.dim(3)):
                        let ix = x * stride - padding + kx
                        let iy = y * stride - padding + ky

                        let kernel_index = (
                            out_ch * kernel.strides[0]
                            + in_ch * kernel.strides[1]
                            + kx * kernel.strides[2]
                            + ky
                        )

                        let input_index = (
                            batch * inputs.strides[0]
                            + in_ch * inputs.strides[1]
                            + ix * inputs.strides[2]
                            + iy
                        )

                        result += (
                            inputs.tensor[input_index] * kernel.tensor[kernel_index]
                        )

            let output_index = (
                batch * outputs_strides[0]
                + out_ch * outputs_strides[1]
                + x * outputs_strides[2]
                + y
            )

            outputs[output_index] = result + bias.tensor[out_ch]

        let oH_border_0 = 0
        let oH_border_1 = (padding + kernel.strides[0] + 1) / kernel.strides[0]
        let oH_border_2 = (
            inputs.tensor.dim(2) + padding - kernel.tensor.dim(2)
        ) / kernel.strides[0]
        let oH_border_3 = outputs.dim(2)

        let oW_border_0 = 0
        let oW_border_1 = (padding + kernel.strides[1] + 1) / kernel.strides[1]
        let oW_border_2 = (
            inputs.tensor.dim(3) + padding - kernel.tensor.dim(3)
        ) / kernel.strides[1]
        let oW_border_3 = outputs.dim(3)

        for batch in range(inputs.tensor.dim(0)):
            for out_ch in range(outputs.dim(1)):
                let batch_o_idx = batch * outputs_strides[0]
                let out_ch_o_idx = out_ch * outputs_strides[1]
                # Case 1: oh might put us out of bounds
                for x in range(oH_border_0, oH_border_1):
                    for y in range(outputs.dim(3)):
                        kernel_iteration_all_checks(batch, out_ch, x, y)
                # Case 2: oh in bounds
                for x in range(oH_border_1, oH_border_2):
                    # Case a: ow might put us out of bounds
                    for y in range(oW_border_0, oW_border_1):
                        kernel_iteration_all_checks(batch, out_ch, x, y)
                    # Case b: ow in bounds
                    for y in range(oW_border_1, oW_border_2):
                        kernel_iteration_no_checks(batch, out_ch, x, y)
                    # Case c: ow might put us out of bounds
                    for y in range(oW_border_2, oW_border_3):
                        kernel_iteration_all_checks(batch, out_ch, x, y)
                # Case 3: oh might put us out of bounds
                for x in range(oH_border_2, oH_border_3):
                    for y in range(outputs.dim(3)):
                        kernel_iteration_all_checks(batch, out_ch, x, y)

        return GRAPH.create_graph_node[Self.backward[padding, stride]](
            outputs, inputs, kernel, bias
        )

    @staticmethod
    fn backward[
        padding: Int, stride: Int
    ](ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[
        dtype
    ]:
        """
        Backward operation of 2D convolution.

        Upper gradient of shape: [batch, out_channels, uX, uY].
        """

        alias nelts: Int = simdwidthof[dtype]()
        let inputs = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
        let inputs_strides = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].strides
        let kernel = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])].tensor
        let kernel_strides = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])].strides
        let bias = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[2])].tensor
        let bias_strides = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[2])].strides

        let ug_strides = calculate_strides(ug.shape())

        if tensor_id == 0:
            # Inputs
            var res = Tensor[dtype](inputs.shape())

            for batch in range(inputs.dim(0)):
                for in_ch in range(inputs.dim(1)):
                    for ix in range(inputs.dim(2)):
                        for iy in range(inputs.dim(3)):
                            var result: SIMD[dtype, 1] = 0
                            for out_ch in range(ug.dim(1)):
                                for kx in range(kernel.dim(2)):
                                    for ky in range(kernel.dim(3)):
                                        let ux = ix * stride - kx + padding
                                        let uy = iy * stride - ky + padding

                                        if (
                                            ux < 0
                                            or uy < 0
                                            or ux >= ug.dim(2)
                                            or uy >= ug.dim(3)
                                        ):
                                            continue

                                        let kernel_index = (
                                            out_ch * kernel_strides[0]
                                            + in_ch * kernel_strides[1]
                                            + (kernel.dim(2) - kx - 1)
                                            * kernel_strides[2]
                                            + (kernel.dim(3) - ky - 1)
                                        )

                                        let ug_index = (
                                            batch * ug_strides[0]
                                            + out_ch * ug_strides[1]
                                            + ux * ug_strides[2]
                                            + uy
                                        )

                                        result += kernel[kernel_index] * ug[ug_index]

                            let input_index = (
                                batch * inputs_strides[0]
                                + in_ch * inputs_strides[1]
                                + ix * inputs_strides[2]
                                + iy
                            )
                            res[input_index] = result

            return res

        elif tensor_id == 1:
            # Kernel
            var res = Tensor[dtype](kernel.shape())

            for in_ch in range(inputs.dim(1)):
                for out_ch in range(ug.dim(1)):
                    for kx in range(kernel.dim(2)):
                        for ky in range(kernel.dim(3)):
                            var result: SIMD[dtype, 1] = 0
                            for batch in range(inputs.dim(0)):
                                for ux in range(ug.dim(2)):
                                    for uy in range(ug.dim(3)):
                                        let ix = kx * stride - padding + ux
                                        let iy = ky * stride - padding + uy

                                        if (
                                            ix < 0
                                            or iy < 0
                                            or ix >= inputs.dim(2)
                                            or iy >= inputs.dim(3)
                                        ):
                                            continue

                                        let input_index = (
                                            batch * inputs_strides[0]
                                            + in_ch * inputs_strides[1]
                                            + ix * inputs_strides[2]
                                            + iy
                                        )

                                        let ug_index = (
                                            batch * ug_strides[0]
                                            + out_ch * ug_strides[1]
                                            + ux * ug_strides[2]
                                            + uy
                                        )

                                        result += inputs[input_index] * ug[ug_index]

                            let kernel_index = (
                                out_ch * kernel_strides[0]
                                + in_ch * kernel_strides[1]
                                + kx * kernel_strides[2]
                                + ky
                            )
                            res[kernel_index] = result

            return res

        else:
            # Bias
            # Sum of upper gradient over batch and X, Y dimensions
            # out_channels == ug.dim(1) == bias.dim(0)
            var res = Tensor[dtype](bias.shape())

            for out_ch in range(ug.dim(1)):
                var sum: SIMD[dtype, 1] = 0
                for batch in range(ug.dim(0)):
                    for ux in range(ug.dim(2)):
                        for uy in range(ug.dim(3)):
                            let ug_index = (
                                batch * ug_strides[0]
                                + out_ch * ug_strides[1]
                                + ux * ug_strides[2]
                                + uy
                            )
                            sum += ug[ug_index]

                res[out_ch] = sum

            return res


# <------------CONV3D------------>


# <------------MAXPOOL2D------------>


# <------------MAXPOOL3D------------>
