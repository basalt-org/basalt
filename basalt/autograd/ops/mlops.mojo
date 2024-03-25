from algorithm import vectorize
from math import exp, pow

from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import (
    elwise_transform
)



struct SIGMOID:
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    @always_inline
    fn sigmoid[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return 1 / (1 + exp(-x))

    @staticmethod
    @always_inline
    fn sidmoid_bw[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return Self.sigmoid(x) * (1 - Self.sigmoid(x))

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of sigmoid."""
        elwise_transform[Self.sigmoid](res, t1)

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of sigmoid."""
        # d(sigmod(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        var res_grad = Tensor[dtype](ug_shape)
        
        @parameter
        fn vec_sigmoid_bw[nelts: Int](idx: Int):
            res_grad.simd_store[nelts](idx,
                Self.sidmoid_bw(t1.simd_load[nelts](idx)) * ug.simd_load[nelts](idx)
            )

        vectorize[vec_sigmoid_bw, nelts](ug_shape.num_elements())

        return res_grad ^
        

struct RELU:
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    @always_inline
    fn relu[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return x if x > 0 else 0    
    
    @staticmethod
    @always_inline
    fn relu_bw[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return 1 if x > 0 else 0
    
    @staticmethod
    fn forward[
        t1_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of relu."""
        elwise_transform[Self.relu](res, t1)

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of relu."""
        # d(relu(x))/dx = 1 if x > 0 else 0. We also give 0 to x = 0 instead of undefined.
        var res_grad = Tensor[dtype](ug_shape)

        @parameter
        fn vec_relu_bw[nelts: Int](idx: Int):
            res_grad.simd_store[nelts](idx,
                Self.relu_bw(t1.simd_load[nelts](idx)) * ug.simd_load[nelts](idx)
            )

        vectorize[vec_relu_bw, nelts](ug_shape.num_elements())

        return res_grad ^


struct TANH:
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    @always_inline
    fn tanh[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    @staticmethod
    @always_inline
    fn tanh_bw[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return 1 - pow(Self.tanh(x), 2)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of tanh."""
        elwise_transform[Self.tanh](res, t1)

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of tanh."""
        # d(tanh(x))/dx = 1 - tanh(x) ** 2
        var res_grad = Tensor[dtype](ug_shape)

        @parameter
        fn vec_tanh_bw[nelts: Int](idx: Int):
            res_grad.simd_store[nelts](idx,
                Self.tanh_bw(t1.simd_load[nelts](idx)) * ug.simd_load[nelts](idx)
            )

        vectorize[vec_tanh_bw, nelts](ug_shape.num_elements())

        return res_grad ^


# struct SOFTMAX:
#     @staticmethod
#     fn softmax[axis: Int](n: Tensor[dtype]) -> Tensor[dtype]:
#         """Softmax operation."""
#         # exp(x_i - max(x_j)) / sum(exp(x_j))
#         var max_val = tmax[dtype, nelts](n, axis)
#         var x_minus_max = elwise_op[dtype, nelts, sub](n, max_val)

#         var exp_res = elwise_transform[dtype, nelts, exp](x_minus_max)
#         var sum_res = tsum[dtype, nelts](exp_res, axis)
#         var res = elwise_op[dtype, nelts, div](exp_res, sum_res)

#         return res

#     @staticmethod
#     fn forward[axis: Int](n: Node[dtype]) -> Node[dtype]:
#         """Forward operation of softmax."""
#         # softmax: exp(x_i) / sum(exp(x_j))
#         # stable softmax: exp(x_i - max(x_j)) / sum(exp(x_j))
#         var softmax_res = Self.softmax[axis](n.tensor)
#         var res = elwise_op[dtype, nelts, div](n.tensor, softmax_res)

#         return GRAPH.create_graph_node[Self.backward[axis]](res, n)

#     @staticmethod
#     fn backward[axis: Int](
#         ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
#     ) -> Tensor[dtype]:
#         """Backward operation of softmax."""
#         pass