from tensor import TensorShape
from math import add, sub, mul, div, log, exp
from algorithm import vectorize
from memory import memcpy

from dainemo.utils.tensorutils import *

"""
Implement forward and backward operations for basic tensor manipulations.
"""


trait BinaryOperator:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        """
        Returns the shape of the result tensor given the shapes of the input tensors.
        """
        ...


trait UnaryOperator:
    @staticmethod
    fn result_shape(t_shape: TensorShape) -> TensorShape:
        """
        Returns the shape of the result tensor given the shape of the input tensor.
        """
        ...


# ----- Binary operators -----
# <------------ADD------------>
@value
struct ADD(BinaryOperator):
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the add operation.
        """
        elwise_op[t1_shape, t2_shape, add](res, t1, t2)

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of element wise addition."""
        # d(x + y) / dx = d(x + y) / dy = 1
        return ug


# <------------SUB------------>
@value
struct SUB(BinaryOperator):
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the subtraction operation.
        """
        elwise_op[t1_shape, t2_shape, sub](res, t1, t2)

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of element wise subtraction."""
        # d(x - y) / dx = 1
        # d(x - y) / dy = -1
        @parameter
        if tensor_id == 0:
            return ug
        else:
            var res_grad = Tensor[dtype](t2_shape)
            elwise_op[mul](res_grad, ug, -1.0)
            return res_grad ^


# <------------MUL------------>
@value
struct MUL(BinaryOperator):
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the multiplication operation.
        """
        elwise_op[t1_shape, t2_shape, mul](res, t1, t2)

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of element wise multiplication."""
        # d(x * y) / dx = y
        # d(x * y) / dy = x
        @parameter
        if tensor_id == 0:
            var res_grad = Tensor[dtype](t1_shape)
            elwise_op[t1_shape, t2_shape, mul](res_grad, ug, t2)
            return res_grad ^
        else:
            var res_grad = Tensor[dtype](t2_shape)
            elwise_op[t2_shape, t1_shape, mul](res_grad, ug, t1)
            return res_grad ^


# <------------DIV------------>
@value
struct DIV(BinaryOperator):
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        t2_shape: TensorShape
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward operation of element wise division.
        """
        elwise_op[t1_shape, t2_shape, div](res, t1, t2)

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of element wise division."""
        # d(x/y) / dx = 1/y
        # d(x/y) / dy = -x/y^2

        @parameter
        if tensor_id == 0:
            var res_grad = Tensor[dtype](t1_shape)
            elwise_op[t1_shape, t2_shape, div](res_grad, ug, t2)
            return res_grad ^
        else:
            alias broadcast = (t1_shape != t2_shape)
            alias is_scalar = (t2_shape == TensorShape(1))
            var res_grad = Tensor[dtype](t2_shape)

            @parameter
            if is_scalar:
                var factor: SIMD[dtype, 1] = - 1.0 / (t2[0] ** 2)
                @parameter
                fn vec_div_bw_scalar[nelts: Int](i: Int):
                    res_grad.simd_store[nelts](i,
                        factor * t1.simd_load[nelts](i) * ug.simd_load[nelts](i)
                    )
                vectorize[vec_div_bw_scalar, nelts](t2_shape.num_elements())

            elif broadcast and not is_scalar:
                alias strides1 = broadcast_calculate_strides(t1_shape, ug_shape)
                alias strides2 = broadcast_calculate_strides(t2_shape, ug_shape)
                @parameter
                fn vec_div_bw_broadcast[netls: Int](i: Int):
                    var index1 = get_real_index[ug_shape](i, strides1)
                    var index2 = get_real_index[ug_shape](i, strides2)
                    res_grad.simd_store[nelts](i,
                        - t1.simd_load[nelts](index1) / (t2.simd_load[nelts](index2) ** 2) * ug.simd_load[nelts](i)
                    )
                vectorize[vec_div_bw_broadcast, nelts](t2_shape.num_elements())

            else:
                @parameter
                fn vec_div_bw[nelts: Int](i: Int):
                    res_grad.simd_store[nelts](i, 
                        - t1.simd_load[nelts](i) / (t2.simd_load[nelts](i) ** 2) * ug.simd_load[nelts](i)
                    )
                vectorize[vec_div_bw, nelts](t2_shape.num_elements())

            return res_grad ^


# <------------DOT------------>
@value
struct DOT(BinaryOperator):
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return TensorShape(t1_shape[0], t2_shape[1])

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the dot operation.
        """
        dot[t1_shape, t2_shape](res, t1, t2)

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of dot product."""

        @parameter
        if tensor_id == 0:
            # dot(ug, t2.T)
            var res_grad = Tensor[dtype](t1_shape)
            dot_transpose_t2[ug_shape, t2_shape](res_grad, ug, t2)
            return res_grad ^
        else:
            # dot(t1.T, ug)
            var res_grad = Tensor[dtype](t2_shape)
            dot_transpose_t1[t1_shape, ug_shape](res_grad, t1, ug)
            return res_grad ^


# ----- Unary operators -----
# <------------EXP------------>
@value
struct EXP(UnaryOperator):
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of exp."""
        elwise_transform[exp](res, t1)

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of exp."""
        # d(exp(x)) / dx = exp(x)
        var res_grad = Tensor[dtype](t1_shape)

        @parameter
        fn vec_exp_bw[nelts: Int](i: Int):
            res_grad.simd_store[nelts](i,
                exp(t1.simd_load[nelts](i)) * ug.simd_load[nelts](i)
            )
        vectorize[vec_exp_bw, nelts](ug_shape.num_elements())
        return res_grad ^


# <------------LOG------------>
@value
struct LOG:
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of exp."""
        elwise_transform[log](res, t1)

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of log."""
        # d(log(x)) / dx = 1 / x
        var res_grad = Tensor[dtype](t1_shape)
        elwise_op[t1_shape, t1_shape, div](res_grad, ug, t1)
        return res_grad ^


# <------------POW------------>
struct POW(BinaryOperator):
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        # t2_shape == TensorShape(1)
        return t1_shape

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """Forward operation of element wise pow."""
        # t2_shape is a graph scalar
        elwise_pow(res, t1, t2[0].to_int())


    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        t2_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of element wise pow."""
        # d(x^y) / dx = y * x^(y-1)
        # d(x^y) / dy = sum( x^y * log(x) )
        var res_grad: Tensor[dtype]
        var a = t2[0].to_int()

        @parameter
        if tensor_id == 0:
            res_grad = Tensor[dtype](t1_shape)
            @parameter
            fn vec_pow_bw_x[nelts: Int](i: Int):
                res_grad.simd_store[nelts](i,
                    a * (t1.simd_load[nelts](i) ** (a - 1)) * ug.simd_load[nelts](i)
                )
            vectorize[vec_pow_bw_x, nelts](t1_shape.num_elements())

        else:
            res_grad = Tensor[dtype](t2_shape)  # t2_shape == TensorShape(1)
            @parameter
            fn vec_pow_bw_y[nelts: Int](i: Int):
                res_grad[0] += (
                    (t1.simd_load[nelts](i) ** a) * log(t1.simd_load[nelts](i)) * ug.simd_load[nelts](i)
                ).reduce_add()

            vectorize[vec_pow_bw_y, nelts](ug_shape.num_elements()) 

        return res_grad ^


# ----- Reduce operators -----
# <------------MEAN------------>
# TODO: include the axis capabilities
@value
struct MEAN(UnaryOperator):
    @staticmethod
    fn result_shape(t_shape: TensorShape) -> TensorShape:
        # This is the result shape for mean without an axis
        return TensorShape(1)  

    @staticmethod
    fn forward[t_shape: TensorShape](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the mean operation.
        """

        var res_simd = tmean(t)
        
        # there is only one value in the result tensor because wher are not using a specific axis to calculate the mean
        res[0] = res_simd

    @staticmethod
    fn backward[
        ug_shape: TensorShape, t_shape: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of mean."""
        # d(mean(t)) / dt = 1 / t.num_elements()
        var res_grad = Tensor[dtype](t_shape)

        var grad: SIMD[dtype, 1] = 1.0 / t_shape.num_elements()

        grad = grad * ug[0] # because ug is a tensor of size 1 when mean is used without an axis

        @parameter
        fn v_mean_d[nelts: Int](i: Int):
            res_grad.simd_store[nelts](i, grad)

        vectorize[v_mean_d, nelts](ug_shape.num_elements())

        return res_grad ^


# # <------------SUM------------>
# struct SUM:
#     @staticmethod
#     fn forward[axis: Int](n: Node[dtype]) -> Node[dtype]:
#         """Forward pass of sum operation: along axis."""
#         var res: Tensor[dtype] = tsum[dtype, nelts](n.tensor, axis=axis)
#         return GRAPH.create_graph_node[Self.backward[axis=axis]](res, n)

#     @staticmethod
#     fn forward(n: Node[dtype]) -> Node[dtype]:
#         """Forward pass of sum operation: all elements."""
#         var res: SIMD[dtype, 1] = tsum[dtype, nelts](n.tensor)
#         var res_tensor = Tensor[dtype](1)
#         res_tensor[0] = res
#         return GRAPH.create_graph_node[Self.backward[axis=-1]](res_tensor, n)

#     @staticmethod
#     fn backward[
#         axis: Int = -1
#     ](ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[
#         dtype
#     ]:
#         """Backward pass of sum operation."""
#         # Expand the upper gradient to the same shape as the input tensor
#         var t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
#         var res = Tensor[dtype](t.shape())
#         fill[dtype, nelts](res, 1.0)

#         return elwise_op[dtype, nelts, mul](res, ug)


# # <------------MAX------------>
# struct MAX:
#     @staticmethod
#     fn forward[axis: Int](n: Node[dtype]) -> Node[dtype]:
#         """Forward pass of max operation: along axis."""
#         alias nelts: Int = simdwidthof[dtype]()
#         var res: Tensor[dtype] = tmax[dtype, nelts](n.tensor, axis=axis)
#         return GRAPH.create_graph_node[Self.backward[axis=axis]](res, n)

#     @staticmethod
#     fn forward(n: Node[dtype]) -> Node[dtype]:
#         """Forward pass of max operation: all elements."""
#         alias nelts: Int = simdwidthof[dtype]()
#         var res: SIMD[dtype, 1] = tmax[dtype, nelts](n.tensor)
#         var res_tensor = Tensor[dtype](1)
#         res_tensor[0] = res
#         return GRAPH.create_graph_node[Self.backward[axis= -1]](res_tensor, n)

#     @staticmethod
#     fn backward[
#         axis: Int = -1
#     ](ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[
#         dtype
#     ]:
#         """Backward pass of max operation."""
#         # This could be changed to something like in tinygrad:
#         # max_1s = CMPEQ(original_tensor, expanded(max_tensor), axis=axis)
#         # sum_max_1s = SUM(max_1s)
#         # div_sum_max_1s = DIV(max_1, sum_max_1s)

#         # The selected element is 1.0, the others are 0.0. And if there are
#         # multiple max values, the gradient is divided by the number of max
#         # values (1/n) for each max value.
#         alias nelts: Int = simdwidthof[dtype]()
#         var t_node = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])]
#         var t = t_node.tensor
#         var strides = calculate_strides(t.shape())
#         var res = Tensor[dtype](t.shape())

#         @parameter
#         if axis == -1:
#             # ug size is 1
#             var max_res = tmax[dtype, nelts](t)
#             var sum_eq: SIMD[dtype, 1] = 0
#             for i in range(t.num_elements()):
#                 if t[i] == max_res:
#                     sum_eq += 1

#             var factor = 1 / sum_eq
#             for i in range(res.num_elements()):
#                 if t[i] == max_res:
#                     res[i] = factor * ug[0]
#         else:
#             # max_res.shape == ug.shape
#             var max_res = tmax[dtype, nelts](t, axis=axis)

#             for i in range(max_res.num_elements()):
#                 var index_base = (i % strides[axis]) + (i // strides[axis]) * (
#                     strides[axis] * t.dim(axis)
#                 )

#                 var count_1s: SIMD[dtype, 1] = 0
#                 # Count the number of values equal to max_res
#                 for j in range(t.dim(axis)):
#                     var index = index_base + j * strides[axis]
#                     if t[index] == max_res[i]:
#                         count_1s += 1
#                 # Divide 1.0 by the number of max values (n) and multiply by upper gradient value
#                 var factor = 1 / count_1s
#                 for j in range(t.dim(axis)):
#                     var index = index_base + j * strides[axis]
#                     if t[index] == max_res[i]:
#                         res[index] = factor * ug[i]

#         return res


# ----- Transform operators -----
# # <---------TRANSPOSE--------->
# struct TRANSPOSE:
#     @staticmethod
#     fn forward(n: Node[dtype]) -> Node[dtype]:
#         """Forward pass of transpose operation."""
#         var res = transpose[dtype, nelts](n.tensor)
#         return GRAPH.create_graph_node[Self.backward](res, n)

#     @staticmethod
#     fn backward(
#         ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
#     ) -> Tensor[dtype]:
#         """No local gradient. Transpose is its own inverse."""
#         return transpose[dtype, nelts](ug)


# <----------FLATTEN---------->
struct FLATTEN(UnaryOperator):
    @staticmethod
    fn result_shape(t_shape: TensorShape) -> TensorShape:
        return TensorShape(t_shape.num_elements())

    @staticmethod
    fn forward[t_shape: TensorShape](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the flatten operation.
        """
        memcpy(res.data(), t.data(), t_shape.num_elements())

    @staticmethod
    fn backward[
        ug_shape: TensorShape, t_shape: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of flatten."""
        var res_grad = Tensor[dtype](ug_shape)
        memcpy(res_grad.data(), ug.data(), ug_shape.num_elements())

        return res_grad ^


# # <----------RESHAPE---------->
# struct RESHAPE:
#     @staticmethod
#     fn forward(n: Node[dtype], new_shape: TensorShape) -> Node[dtype]:
#         var res = n.tensor
#         try:
#             res.ireshape(new_shape)
#         except:
#             print("[ERROR]: Cannot reshape tensor in forward pass.")

#         return GRAPH.create_graph_node[Self.backward](res, n)

#     @staticmethod
#     fn backward(
#         ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
#     ) -> Tensor[dtype]:
#         """
#         Reshape upper gradient to original shape.
#         """
#         var res = ug
#         var shape = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor.shape()

#         try:
#             res.ireshape(shape)
#         except:
#             print("[ERROR]: Cannot reshape tensor in reshape backward pass.")

#         return res
