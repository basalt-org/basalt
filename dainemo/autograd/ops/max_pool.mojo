from tensor import Tensor, TensorShape
from math import floor

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.autograd.ops.conv import get_result_shape


# <------------MAXPOOL2D------------>
struct MAXPOOL2D:
    @staticmethod
    fn forward[
        padding: Int,
        stride: Int
    ](inputs: Node[dtype], kernel_shape: TensorShape) -> Node[dtype]:
        """
        Returns the max value of each kernel in the input tensor.
            inputs.shape     [batch_size, in_channels, iX, iY]
            kernel.shape     [out_channels, in_channels, kX, kY] 
            outputs.shape    [batch_size, out_channels, oX, oY]
            and for maxpool2d (in_channels == out_channels).
        """
        # TODO: calculate indeces using precalculated strides

        alias nelts: Int = simdwidthof[dtype]()
        
        let result_shape = get_result_shape[padding, stride](
            inputs.tensor.shape(), kernel_shape
        )

        var outputs = Tensor[dtype](
            inputs.tensor.dim(0), kernel_shape[0], result_shape[0], result_shape[1]
        )

        for batch in range(inputs.tensor.dim(0)):
            for in_ch in range(inputs.tensor.dim(1)):
                for x in range(outputs.dim(2)):
                    for y in range(outputs.dim(3)):
                        var max_val: SIMD[dtype, 1] = -1e9
                        for kx in range(kernel_shape[2]):
                            for ky in range(kernel_shape[3]):
                                let ix = x * stride - padding + kx
                                let iy = y * stride - padding + ky

                                if ix < 0 or iy < 0 or ix >= inputs.tensor.dim(2) or iy >= inputs.tensor.dim(3):
                                    continue

                                let max_idx = (
                                    batch * (inputs.tensor.dim(1) * inputs.tensor.dim(2) * inputs.tensor.dim(3)) 
                                    + in_ch * (inputs.tensor.dim(2) * inputs.tensor.dim(3))
                                    + ix * inputs.tensor.dim(3) 
                                    + iy
                                )

                                let val = inputs.tensor[max_idx]
                                if val > max_val:
                                    max_val = val

                        let out_idx = (
                            batch * (outputs.dim(1) * outputs.dim(2) * outputs.dim(3)) 
                            + in_ch * (outputs.dim(2) * outputs.dim(3))
                            + x * outputs.dim(3) 
                            + y
                        )

                        outputs[out_idx] = max_val

        return GRAPH.create_graph_node[Self.backward[padding, stride]](outputs, inputs)

    @staticmethod
    fn backward[
        padding: Int, stride: Int
    ](
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """
        Backward operation of MAXPOOL2D.

        Upper gradient of shape: [batch_size, out_channels, uX, uY]
        """
        # TODO

        return Tensor[dtype]()


# <------------MAXPOOL3D------------>
# TODO