from basalt import Tensor, TensorShape
from basalt.autograd.attributes import AttributeVector
from basalt.utils.tensorutils import dot, dot_transpose_t1, dot_transpose_t2

from algorithm import parallelize, vectorize
from math import divmod
from utils.loop import unroll


@always_inline
fn get_result_shape(
    input_shape: TensorShape,
    kernel_shape: TensorShape,
    padding: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
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


struct CONV2D:
    @staticmethod
    fn result_shape(
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector,
    ) -> TensorShape:
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
        attributes: AttributeVector,
    ](
        inout outputs: Tensor[dtype],
        inputs: Tensor[dtype],
        kernel: Tensor[dtype],
        bias: Tensor[dtype],
    ):
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

        alias padding_x = padding[0]
        alias padding_y = padding[1]
        alias stride_x = stride[0]
        alias stride_y = stride[1]
        alias dilation_x = dilation[0]
        alias dilation_y = dilation[1]

        alias batch_size = input_shape[0]
        alias in_channels = input_shape[1]
        alias in_x = input_shape[2]
        alias in_y = input_shape[3]
        alias out_channels = kernel_shape[0]
        alias k_x = kernel_shape[2]
        alias k_y = kernel_shape[3]
        alias out_x = output_shape[2]
        alias out_y = output_shape[3]
        alias col_x = out_x
        alias col_y = out_y

        alias col_shape = TensorShape(
            batch_size, col_x * col_y, in_channels * k_x * k_y
        )  # [batch, colX * colY, in_channels * kX * kY]
        alias output_shape = Self.result_shape(
            input_shape, kernel_shape, bias_shape, attributes
        )
        alias col_shape_stripped = TensorShape(in_channels * k_x * k_y, col_x, col_y)

        alias inputs_strides = input_shape.strides()
        alias kernel_strides = kernel_shape.strides()
        alias outputs_strides = output_shape.strides()
        alias col_strides = col_shape.strides()

        var col_ptr = DTypePointer[dtype].alloc(col_shape.num_elements())
        memset_zero(col_ptr, col_shape.num_elements())

        @parameter
        fn im2col(batch: Int):
            for ux in range(out_x):
                for uy in range(out_y):
                    for in_ch in range(in_channels):
                        for kx in range(k_x):
                            for ky in range(k_y):
                                var ix = ux * stride_x - padding_x + kx * dilation_x
                                var iy = uy * stride_y - padding_y + ky * dilation_y

                                if ix < 0 or iy < 0 or ix >= in_x or iy >= in_y:
                                    continue

                                var col_index = (
                                    batch * col_strides[0]
                                    + (ux * col_y + uy) * col_strides[1]
                                    + (in_ch * k_x * k_y + kx * k_y + ky)
                                )

                                var input_index = (
                                    batch * inputs_strides[0]
                                    + in_ch * inputs_strides[1]
                                    + ix * inputs_strides[2]
                                    + iy
                                )

                                col_ptr[col_index] = inputs[input_index]

        parallelize[im2col](batch_size)

        @parameter
        fn conv(batch: Int):
            for out_ch in range(out_channels):
                for ux in range(out_x):
                    for uy in range(out_y):
                        var result: SIMD[dtype, nelts] = 0

                        @parameter
                        fn v_im2col[_nelts: Int](in_ch_kx_ky: Int):
                            var col_index = (
                                batch * col_strides[0]
                                + (ux * col_y + uy) * col_strides[1]
                                + in_ch_kx_ky
                            )

                            var kernel_index = (
                                out_ch * kernel_strides[0] + in_ch_kx_ky
                            )

                            @parameter
                            if _nelts == nelts:
                                result += col_ptr.load[width=nelts](
                                    col_index
                                ) * kernel.load[nelts](kernel_index)
                            else:
                                result[0] += (
                                    col_ptr.load[width=_nelts](col_index)
                                    * kernel.load[_nelts](kernel_index)
                                ).reduce_add()

                        vectorize[v_im2col, nelts](in_channels * k_x * k_y)

                        var output_index = (
                            batch * outputs_strides[0]
                            + out_ch * outputs_strides[1]
                            + ux * outputs_strides[2]
                            + uy
                        )

                        outputs[output_index] = result.reduce_add() + bias[out_ch]

        parallelize[conv](batch_size)

        col_ptr.free()

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        input_shape: TensorShape,
        kernel_shape: TensorShape,
        bias_shape: TensorShape,
        attributes: AttributeVector,
    ](
        ug: Tensor[dtype],
        inputs: Tensor[dtype],
        kernel: Tensor[dtype],
        bias: Tensor[dtype],
    ) -> Tensor[dtype]:
        """
        Backward operation of 2D convolution.

        Upper gradient of shape: [batch, out_channels, uX, uY].
        """

        alias padding = attributes["padding"].value().to_static[2]()
        alias stride = attributes["stride"].value().to_static[2]()
        alias dilation = attributes["dilation"].value().to_static[2]()
        alias padding_0 = padding[0]
        alias padding_1 = padding[1]
        alias stride_0 = stride[0]
        alias stride_1 = stride[1]
        alias dilation_0 = dilation[0]
        alias dilation_1 = dilation[1]

        alias inputs_strides = input_shape.strides()
        alias kernel_strides = kernel_shape.strides()
        alias ug_strides = ug_shape.strides()
        alias inputs_strides_0 = inputs_strides[0]
        alias inputs_strides_1 = inputs_strides[1]
        alias inputs_strides_2 = inputs_strides[2]
        alias kernel_strides_0 = kernel_strides[0]
        alias kernel_strides_1 = kernel_strides[1]
        alias kernel_strides_2 = kernel_strides[2]
        alias ug_strides_0 = ug_strides[0]
        alias ug_strides_1 = ug_strides[1]
        alias ug_strides_2 = ug_strides[2]

        alias input_shape_0 = input_shape[0]
        alias input_shape_1 = input_shape[1]
        alias input_shape_2 = input_shape[2]
        alias input_shape_3 = input_shape[3]
        alias kernel_shape_2 = kernel_shape[2]
        alias kernel_shape_3 = kernel_shape[3]
        alias ug_shape_0 = ug_shape[0]
        alias ug_shape_1 = ug_shape[1]
        alias ug_shape_2 = ug_shape[2]
        alias ug_shape_3 = ug_shape[3]

        var res: Tensor[dtype]

        @parameter
        if tensor_id == 0:
            # Inputs
            # Sum of upper gradient over batch, X, Y dimensions
            # In lament terms, for every element in the input tensor, add the sum of every element in the kernel times the corresponding element in the upper gradient.
            res = Tensor[dtype](input_shape)
            
            @parameter
            fn input_grad(batch: Int):
                var batch_offset = batch * inputs_strides_0
                for i in range(input_shape_1 * input_shape_2 * input_shape_3):
                    var input_index = batch_offset + i
                    var ug_val = ug[i // (kernel_shape_2 * kernel_shape_3)]

                    @parameter
                    fn vec_kernel[Nelts: Int](index: Int):
                        res[input_index] += (
                            kernel.load[simd_width=nelts](index) * ug_val
                        ).reduce_add()

                    vectorize[
                        vec_kernel, nelts, size = kernel_shape_2 * kernel_shape_3
                    ]()

            parallelize[input_grad](input_shape_0)

        elif tensor_id == 1:
            # Kernel
            # Sum of upper gradient over batch and X, Y dimensions
            res = Tensor[dtype](kernel_shape)

            @parameter
            fn kernel_grad(in_ch: Int):
                for out_ch in range(ug_shape_1):
                    for kx in range(kernel_shape_2):
                        for ky in range(kernel_shape_3):
                            var result: SIMD[dtype, 1] = 0
                            for batch in range(input_shape_0):
                                for ux in range(ug_shape_2):
                                    for uy in range(ug_shape_3):
                                        var ix = ux * stride_0 - padding_0 + kx * dilation_0
                                        var iy = uy * stride_1 - padding_1 + ky * dilation_1

                                        if (
                                            ix < 0
                                            or iy < 0
                                            or ix >= input_shape_2
                                            or iy >= input_shape_3
                                        ):
                                            continue

                                        var input_index = (
                                            batch * inputs_strides_0
                                            + in_ch * inputs_strides_1
                                            + ix * inputs_strides_2
                                            + iy
                                        )

                                        var ug_index = (
                                            batch * ug_strides_0
                                            + out_ch * ug_strides_1
                                            + ux * ug_strides_2
                                            + uy
                                        )

                                        result += inputs[input_index] * ug[ug_index]

                            var kernel_index = (
                                out_ch * kernel_strides_0
                                + in_ch * kernel_strides_1
                                + kx * kernel_strides_2
                                + ky
                            )
                            res[kernel_index] = result

            parallelize[kernel_grad](input_shape_1)

        else:
            # Bias
            # Sum of upper gradient over batch and X, Y dimensions
            # out_channels == ug_shape[1] == bias_shape[0]
            res = Tensor[dtype](bias_shape)

            @parameter
            fn bias_grad(out_ch: Int):
                var sum: SIMD[dtype, 1] = 0
                for batch in range(ug_shape_0):
                    for ux in range(ug_shape_2):
                        for uy in range(ug_shape_3):
                            var ug_index = (
                                batch * ug_strides_0
                                + out_ch * ug_strides_1
                                + ux * ug_strides_2
                                + uy
                            )
                            sum += ug[ug_index]

                res[out_ch] = sum

            parallelize[bias_grad](ug_shape_1)

        return res
