from math.limit import neginf

from basalt import Tensor, TensorShape
from basalt.autograd.attributes import AttributeVector
from basalt.autograd.ops.conv import get_result_shape



struct MAXPOOL2D:
    @staticmethod
    fn result_shape(input_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        
        var kernel_size = attributes["kernel_size"].value().to_static[2]()
        var padding = attributes["padding"].value().to_static[2]()
        var stride = attributes["stride"].value().to_static[2]()
        var dilation = attributes["dilation"].value().to_static[2]()
        
        var res = get_result_shape(input_shape, TensorShape(kernel_size[0], kernel_size[1]), padding, stride, dilation)

        return TensorShape(input_shape[0], input_shape[1], res[0], res[1])
    
    @staticmethod
    fn forward[
        input_shape: TensorShape,
        attributes: AttributeVector
    ](inout outputs: Tensor[dtype], inputs: Tensor[dtype]):
        """
        Returns the max value of each kernel in the input tensor.
            inputs.shape     [batch_size, channels, iX, iY]
            with kernel_size = (kX, kY)
            outputs.shape    [batch_size, channels, oX, oY].
        """
        alias kernel_size = attributes["kernel_size"].value().to_static[2]()
        alias padding = attributes["padding"].value().to_static[2]()
        alias stride = attributes["stride"].value().to_static[2]()
        alias dilation = attributes["dilation"].value().to_static[2]()

        alias inputs_strides = input_shape.strides()
        alias output_shape = Self.result_shape(input_shape, attributes)
        alias outputs_strides = output_shape.strides()

        for batch in range(input_shape[0]):
            for in_ch in range(input_shape[1]):
                for x in range(output_shape[2]):
                    for y in range(output_shape[3]):
                        var max_val: SIMD[dtype, 1] = neginf[dtype]()
                        var ix_base = x * stride[0] - padding[0]
                        var iy_base = y * stride[1] - padding[1]
                        for kx in range(kernel_size[0]):
                            for ky in range(kernel_size[1]):
                                var ix = ix_base + kx * dilation[0]
                                var iy = iy_base + ky * dilation[1]

                                if (
                                    ix < 0
                                    or iy < 0
                                    or ix >= input_shape[2]
                                    or iy >= input_shape[3]
                                ):
                                    continue

                                var idx = (
                                    batch * inputs_strides[0]
                                    + in_ch * inputs_strides[1]
                                    + ix * inputs_strides[2]
                                    + iy
                                )

                                var val = inputs[idx]
                                if val > max_val:
                                    max_val = val

                        var out_idx = (
                            batch * outputs_strides[0]
                            + in_ch * outputs_strides[1]
                            + x * outputs_strides[2]
                            + y
                        )

                        outputs[out_idx] = max_val

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        input_shape: TensorShape,
        attributes: AttributeVector
    ](ug: Tensor[dtype], inputs: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of MAXPOOL2D.

        Upper gradient of shape: [batch_size, channels, uX, uY]
        """
        alias kernel_size = attributes["kernel_size"].value().to_static[2]()
        alias padding = attributes["padding"].value().to_static[2]()
        alias stride = attributes["stride"].value().to_static[2]()
        alias dilation = attributes["dilation"].value().to_static[2]()

        alias ug_strides = ug_shape.strides()
        alias inputs_strides = input_shape.strides()

        var res = Tensor[dtype](input_shape)

        for batch in range(input_shape[0]):
            for in_ch in range(input_shape[1]):
                for x in range(ug_shape[2]):
                    for y in range(ug_shape[3]):
                        var max_val: SIMD[dtype, 1] = neginf[dtype]()
                        var max_idx: Int = -1
                        var ix_base = x * stride[0] - padding[0]
                        var iy_base = y * stride[1] - padding[1]
                        for kx in range(kernel_size[0]):
                            for ky in range(kernel_size[1]):
                                var ix = ix_base + kx * dilation[0]
                                var iy = iy_base + ky * dilation[1]
                                
                                if (
                                    ix < 0
                                    or iy < 0
                                    or ix >= input_shape[2]
                                    or iy >= input_shape[3]
                                ):
                                    continue

                                var idx = (
                                    batch * inputs_strides[0]
                                    + in_ch * inputs_strides[1]
                                    + ix * inputs_strides[2]
                                    + iy
                                )

                                var val = inputs[idx]
                                if val > max_val:
                                    max_val = val
                                    max_idx = idx

                        var ug_idx = (
                            batch * ug_strides[0] 
                            + in_ch * ug_strides[1]
                            + x * ug_strides[2]
                            + y
                        )

                        res[max_idx] += ug[ug_idx]

        return res