from tensor import TensorShape
from math.limit import neginf

from dainemo.autograd.attributes import AttributeVector
from dainemo.autograd.ops.conv import get_result_shape
from dainemo.utils.tensorutils import calculate_strides



# <------------MAXPOOL2D------------>
struct MAXPOOL2D:
    @staticmethod
    fn result_shape(input_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        
        var kernel_size = attributes["kernel_size"].value().to_static[2]()
        var padding = attributes["padding"].value().to_static[2]()
        var stride = attributes["stride"].value().to_static[2]()
        var dilation = attributes["dilation"].value().to_static[2]()
        
        var res = get_result_shape(input_shape, TensorShape(kernel_size), padding, stride, dilation)

        return TensorShape(input_shape[0], input_shape[1], res[0], res[1])
    
    @staticmethod
    fn forward[
        input_shape: TensorShape,
        attributes: AttributeVector
    ](inout outputs: Tensor[dtype], inputs: Tensor[dtype]):
        """
        Returns the max value of each kernel in the input tensor.
            inputs.shape     [batch_size, in_channels, iX, iY]
            with kernel_size = (kX, kY)
            outputs.shape    [batch_size, in_channels, oX, oY].
        """
        alias kernel_size = attributes["kernel_size"].value().to_static[2]()
        alias padding = attributes["padding"].value().to_static[2]()
        alias stride = attributes["stride"].value().to_static[2]()
        alias dilation = attributes["dilation"].value().to_static[2]()

        alias inputs_strides = calculate_strides(input_shape)
        alias output_shape = Self.result_shape(input_shape, attributes)
        alias outputs_strides = calculate_strides(output_shape)

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

#     @staticmethod
#     fn backward[
#         kernel_size: StaticIntTuple[2],
#         padding: StaticIntTuple[2],
#         stride: StaticIntTuple[2],
#         dilation: StaticIntTuple[2]
#     ](
#         ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
#     ) -> Tensor[dtype]:
#         """
#         Backward operation of MAXPOOL2D.

#         Upper gradient of shape: [batch_size, out_channels, uX, uY]
#         """
#         var inputs = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
#         var inputs_strides = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].strides
#         var res = Tensor[dtype](inputs.shape())
        
#         var ug_strides = calculate_strides(ug.shape())

#         for batch in range(inputs.dim(0)):
#             for in_ch in range(inputs.dim(1)):
#                 for x in range(ug.dim(2)):
#                     for y in range(ug.dim(3)):
#                         var max_val: SIMD[dtype, 1] = neginf[dtype]()
#                         var max_idx: Int = -1
#                         var ix_base = x * stride[0] - padding[0]
#                         var iy_base = y * stride[1] - padding[1]
#                         for kx in range(kernel_size[0]):
#                             for ky in range(kernel_size[1]):
#                                 var ix = ix_base + kx * dilation[0]
#                                 var iy = iy_base + ky * dilation[1]
                                
#                                 if (
#                                     ix < 0
#                                     or iy < 0
#                                     or ix >= inputs.dim(2)
#                                     or iy >= inputs.dim(3)
#                                 ):
#                                     continue

#                                 var idx = (
#                                     batch * inputs_strides[0]
#                                     + in_ch * inputs_strides[1]
#                                     + ix * inputs_strides[2]
#                                     + iy
#                                 )

#                                 var val = inputs[idx]
#                                 if val > max_val:
#                                     max_val = val
#                                     max_idx = idx

#                         var ug_idx = (
#                             batch * ug_strides[0] 
#                             + in_ch * ug_strides[1]
#                             + x * ug_strides[2]
#                             + y
#                         )

#                         res[max_idx] += ug[ug_idx]

#         return res


# # <------------MAXPOOL3D------------>
# # TODO