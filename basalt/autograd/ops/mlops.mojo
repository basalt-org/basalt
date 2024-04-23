from algorithm import vectorize
from math import exp, pow
from math.limit import min_finite, max_finite

from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import elwise_transform
from basalt.autograd.attributes import Attribute, AttributeVector


@value
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
            res_grad.store[nelts](
                idx,
                Self.sidmoid_bw(t1.load[nelts](idx)) * ug.load[nelts](idx),
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
        # x if x > 0 else 0
        return (x > 0).select(x, 0)

    @staticmethod
    @always_inline
    fn relu_bw[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        # 1 if x > 0 else 0
        return (x > 0).select[type](1, 0)

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
            res_grad.store[nelts](
                idx, Self.relu_bw(t1.load[nelts](idx)) * ug.load[nelts](idx)
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
            res_grad.store[nelts](
                idx, Self.tanh_bw(t1.load[nelts](idx)) * ug.load[nelts](idx)
            )

        vectorize[vec_tanh_bw, nelts](ug_shape.num_elements())

        return res_grad ^


struct CLIP:
    @staticmethod
    fn result_shape(t_shape: TensorShape) -> TensorShape:
        return t_shape

    @staticmethod
    fn forward[
        t_shape: TensorShape, attributes: AttributeVector
    ](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the clip operation.
        """
        alias min_attr = attributes["min"]
        alias max_attr = attributes["max"]

        var min_val = min_attr.value().to_scalar[dtype]() if min_attr else min_finite[
            dtype
        ]()
        var max_val = max_attr.value().to_scalar[dtype]() if max_attr else max_finite[
            dtype
        ]()

        @parameter
        fn vec_clip[nelts: Int](i: Int):
            res.store[nelts](i, t.load[nelts](i).min(max_val).max(min_val))

        vectorize[vec_clip, nelts, size = t_shape.num_elements()]()

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t_shape: TensorShape,
        attributes: AttributeVector = AttributeVector(),
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of clip."""
        alias min_attr = attributes["min"]
        alias max_attr = attributes["max"]

        var min_val = min_attr.value().to_scalar[dtype]() if min_attr else min_finite[
            dtype
        ]()
        var max_val = max_attr.value().to_scalar[dtype]() if max_attr else max_finite[
            dtype
        ]()

        var res_grad = Tensor[dtype](t_shape)

        @parameter
        fn vec_clip_bw[nelts: Int](i: Int):
            var val = t.load[nelts](i)
            res_grad.store[nelts](
                i,
                ((val >= min_val) * (val <= max_val)).select(ug.load[nelts](i), 0),
            )

        vectorize[vec_clip_bw, nelts, size = t_shape.num_elements()]()

        return res_grad ^


struct SQUEEZE:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var dim = attributes["dims"]
        var dims_to_squeeze = dim.value().to_shape() if dim else TensorShape()

        var new_shape = List[Int]()
        for i in range(t1_shape.rank()):
            if (not dim and t1_shape[i] == 1) or (i in dims_to_squeeze and t1_shape[i] == 1): 
                continue
            new_shape.append(t1_shape[i])

        return TensorShape(new_shape)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        memcpy(res.data(), t1.data(), t1.num_elements())

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        var res_grad = Tensor[dtype](t1_shape)
        memcpy(res_grad.data(), ug.data(), ug.num_elements())
        return res_grad ^


struct UNSQUEEZE:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var dim = attributes["dims"]
        var dims_to_squeeze = dim.value().to_shape() if dim else TensorShape()

        # Position in the expanded dims where the new dim (or dims) is placed.
        var new_rank = t1_shape.rank() + dims_to_squeeze.rank()

        var new_shape = List[Int]()
        var j = 0
        for i in range(new_rank):
            if i in dims_to_squeeze or i - new_rank in dims_to_squeeze:
                new_shape.append(1)
            else:
                new_shape.append(t1_shape[j])
                j += 1

        return TensorShape(new_shape)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        memcpy(res.data(), t1.data(), t1.num_elements())

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        var res_grad = Tensor[dtype](t1_shape)
        memcpy(res_grad.data(), ug.data(), ug.num_elements())
        return res_grad ^


struct CONCAT2:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        # Assumptions: all tensors have the same shape, except for the dimension specified, or are empty
        var dim = attributes["dims"].value().to_int() if attributes["dims"] else 0

        var res_shape = List[Int]()

        for i in range(t1_shape.rank()):
            if i == dim:
                res_shape.append(t1_shape[i] + t2_shape[i])
            else:
                res_shape.append(t1_shape[i])

        return TensorShape(res_shape)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        t2_shape: TensorShape,
        attributes: AttributeVector,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """Forward operation of concat."""
        alias res_shape = Self.result_shape(t1_shape, t2_shape, attributes)
        var dim = attributes["dims"].value().to_int() if attributes["dims"] else 0

        var t1_size = t1_shape[dim]
        var t2_size = t2_shape[dim]

        memcpy(res.data(), t1.data(), t1.num_elements())
        memcpy(res.data() + t1.num_elements(), t2.data(), t2.num_elements())

    @staticmethod
    fn backward[
        tensor_id: Int,
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        t2_shape: TensorShape,
        attributes: AttributeVector,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of concat."""
        var dim = attributes["dims"].value().to_int() if attributes["dims"] else 0

        @parameter
        if tensor_id == 0:
            var t1_size = t1_shape[dim]
            var t1_grad = Tensor[dtype](t1_shape)
            memcpy(t1_grad.data(), ug.data(), t1.num_elements())
            return t1_grad ^
        else:
            var t2_size = t2_shape[dim]
            var t2_grad = Tensor[dtype](t2_shape)
            memcpy(t2_grad.data(), ug.data() + t1.num_elements(), t2.num_elements())
            return t2_grad ^
