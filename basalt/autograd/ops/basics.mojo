from math import add, sub, mul, div, log, exp
from algorithm import vectorize
from memory import memcpy

from basalt import Tensor, TensorShape
from basalt.nn.tensor import max_rank
from basalt.utils.tensorutils import (
    broadcast_calculate_strides,
    broadcast_shapes,
    dot_transpose_t1,
    dot_transpose_t2,
    elwise_transform,
    get_reduce_shape,
    get_real_index,
    elwise_pow,
    transpose,
    elwise_op,
    tsum,
    fill,
    tmean,
    tmax,
    dot,
)
from basalt.autograd.attributes import Attribute, AttributeVector


@register_passable("trivial")
struct Add:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the add operation.
        """
        elwise_op[Shape1, Shape2, add](res, t1, t2)

    @staticmethod
    fn backward[
        TensorID: Int,
        ShapeUG: TensorShape,
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of element wise addition.
        """
        # d(x + y) / dx = d(x + y) / dy = 1
        return ug


@register_passable("trivial")
struct Sub:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the subtraction operation.
        """
        elwise_op[Shape1, Shape2, sub](res, t1, t2)

    @staticmethod
    fn backward[
        TensorID: Int,
        ShapeUG: TensorShape,
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of element wise subtraction.
        """

        # d(x - y) / dx = 1
        # d(x - y) / dy = -1
        @parameter
        if TensorID == 0:
            return ug
        else:
            var res_grad = Tensor[dtype](ShapeUG)
            elwise_op[mul](res_grad, ug, -1.0)
            return res_grad ^


@register_passable("trivial")
struct Mul:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the multiplication operation.
        """
        elwise_op[Shape1, Shape2, mul](res, t1, t2)

    @staticmethod
    fn backward[
        TensorID: Int,
        ShapeUG: TensorShape,
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of element wise multiplication.
        """

        # d(x * y) / dx = y
        # d(x * y) / dy = x
        @parameter
        if TensorID == 0:
            var res_grad = Tensor[dtype](ShapeUG)
            elwise_op[ShapeUG, Shape2, mul](res_grad, ug, t2)
            return res_grad ^
        else:
            var res_grad = Tensor[dtype](ShapeUG)
            elwise_op[ShapeUG, Shape1, mul](res_grad, ug, t1)
            return res_grad ^


@register_passable("trivial")
struct Div:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return broadcast_shapes(t1_shape, t2_shape)

    @staticmethod
    fn forward[
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward operation of element wise division.
        """
        elwise_op[Shape1, Shape2, div](res, t1, t2)

    @staticmethod
    fn backward[
        TensorID: Int,
        ShapeUG: TensorShape,
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of element wise division.
        """
        # d(x/y) / dx = 1/y
        # d(x/y) / dy = -x/y^2

        @parameter
        if TensorID == 0:
            var res_grad = Tensor[dtype](ShapeUG)
            elwise_op[ShapeUG, Shape2, div](res_grad, ug, t2)
            return res_grad ^
        else:
            alias broadcast = (Shape1 != Shape2)
            alias is_scalar = (Shape2 == TensorShape(1))
            var res_grad = Tensor[dtype](ShapeUG)

            @parameter
            if is_scalar:
                var factor: SIMD[dtype, 1] = -1.0 / (t2[0] ** 2)

                @parameter
                fn vec_div_bw_scalar[nelts: Int](i: Int):
                    res_grad.store[nelts](
                        i, factor * t1.load[nelts](i) * ug.load[nelts](i)
                    )

                vectorize[vec_div_bw_scalar, nelts](ShapeUG.num_elements())

            elif broadcast and not is_scalar:
                alias size = ShapeUG.rank()
                alias strides1 = broadcast_calculate_strides[size, Shape1, ShapeUG]()
                alias strides2 = broadcast_calculate_strides[size, Shape2, ShapeUG]()

                @parameter
                fn vec_div_bw_broadcast[netls: Int](i: Int):
                    var index1 = get_real_index[size, strides1, ShapeUG](i)
                    var index2 = get_real_index[size, strides2, ShapeUG](i)
                    res_grad.store[nelts](
                        i,
                        -t1.load[nelts](index1)
                        / (t2.load[nelts](index2) ** 2)
                        * ug.load[nelts](i),
                    )

                vectorize[vec_div_bw_broadcast, 1](ShapeUG.num_elements())

            else:

                @parameter
                fn vec_div_bw[nelts: Int](i: Int):
                    res_grad.store[nelts](
                        i,
                        -t1.load[nelts](i)
                        / (t2.load[nelts](i) ** 2)
                        * ug.load[nelts](i),
                    )

                vectorize[vec_div_bw, nelts](ShapeUG.num_elements())

            return res_grad ^


@register_passable("trivial")
struct Dot:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        return TensorShape(t1_shape[0], t2_shape[1])

    @staticmethod
    fn forward[
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward pass of the dot operation.
        """
        dot[Shape1, Shape2](res, t1, t2)

    @staticmethod
    fn backward[
        TensorID: Int,
        ShapeUG: TensorShape,
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of dot product.
        """

        @parameter
        if TensorID == 0:
            # dot(ug, t2.T)
            var res_grad = Tensor[dtype](Shape1)
            dot_transpose_t2[ShapeUG, Shape2](res_grad, ug, t2)
            return res_grad ^
        else:
            # dot(t1.T, ug)
            var res_grad = Tensor[dtype](Shape2)
            dot_transpose_t1[Shape1, ShapeUG](res_grad, t1, ug)
            return res_grad ^


@register_passable("trivial")
struct Exp:
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    fn forward[
        Shape1: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of exp."""
        elwise_transform[exp](res, t1)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape,
        Shape1: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of exp.
        """
        # d(exp(x)) / dx = exp(x)
        var res_grad = Tensor[dtype](ShapeUG)

        @parameter
        fn vec_exp_bw[nelts: Int](i: Int):
            res_grad.store[nelts](i, exp(t1.load[nelts](i)) * ug.load[nelts](i))

        vectorize[vec_exp_bw, nelts](ShapeUG.num_elements())
        return res_grad ^


@register_passable("trivial")
struct Log:
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    fn forward[
        Shape1: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """
        Forward operation of exp.
        """
        elwise_transform[log](res, t1)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape,
        Shape1: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of log.
        """
        # d(log(x)) / dx = 1 / x
        var res_grad = Tensor[dtype](ShapeUG)
        elwise_op[ShapeUG, Shape1, div](res_grad, ug, t1)
        return res_grad ^


@register_passable("trivial")
struct Pow:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, t2_shape: TensorShape) -> TensorShape:
        # t2_shape == TensorShape(1)
        return t1_shape

    @staticmethod
    fn forward[
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        """
        Forward operation of element wise pow.
        """
        # t2_shape is a graph scalar
        elwise_pow(res, t1, t2[0].to_int())

    @staticmethod
    fn backward[
        TensorID: Int,
        ShapeUG: TensorShape,
        Shape1: TensorShape,
        Shape2: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of element wise pow.
        """
        # d(x^y) / dx = y * x^(y-1)
        # d(x^y) / dy = sum( x^y * log(x) )
        var res_grad: Tensor[dtype]
        var a = t2[0].to_int()

        @parameter
        if TensorID == 0:
            res_grad = Tensor[dtype](Shape1)

            @parameter
            fn vec_pow_bw_x[nelts: Int](i: Int):
                res_grad.store[nelts](
                    i, a * (t1.load[nelts](i) ** (a - 1)) * ug.load[nelts](i)
                )

            vectorize[vec_pow_bw_x, nelts](Shape1.num_elements())

        else:
            res_grad = Tensor[dtype](Shape2)  # t2_shape == TensorShape(1)

            @parameter
            fn vec_pow_bw_y[nelts: Int](i: Int):
                res_grad[0] += (
                    (t1.load[nelts](i) ** a)
                    * log(t1.load[nelts](i))
                    * ug.load[nelts](i)
                ).reduce_add()

            vectorize[vec_pow_bw_y, nelts](ShapeUG.num_elements())

        return res_grad ^


@register_passable("trivial")
struct Sum:
    @staticmethod
    fn result_shape(t_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var axis = attributes["axis"]

        if axis:
            return get_reduce_shape(t_shape, axis.value().to_int())
        else:
            return TensorShape(1)

    @staticmethod
    fn forward[
        ShapeT: TensorShape, Attributes: AttributeVector
    ](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the sum operation.
        """

        alias axis = Attributes["axis"]

        @parameter
        if axis:
            tsum(res, t, axis.value().to_int())
        else:
            res[0] = tsum(t)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape, Attributes: AttributeVector
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of sum.
        """
        return Self.backward[ShapeUG, ShapeT](ug, t)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of sum.
        """
        var res_grad = Tensor[dtype](ShapeT)
        fill(res_grad, 1.0)

        elwise_op[ShapeT, ShapeUG, mul](res_grad, res_grad, ug)

        return res_grad ^


@register_passable("trivial")
struct Mean:
    @staticmethod
    fn result_shape(t_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var axis = attributes["axis"]

        if axis:
            return get_reduce_shape(t_shape, axis.value().to_int())
        else:
            return TensorShape(1)

    @staticmethod
    fn forward[
        ShapeT: TensorShape, Attributes: AttributeVector
    ](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the mean operation.
        """

        alias axis = Attributes["axis"]

        @parameter
        if axis:
            tmean(res, t, axis.value().to_int())
        else:
            res[0] = tmean(t)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape, Attributes: AttributeVector
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of mean.
        """

        alias axis = Attributes["axis"]

        @parameter
        if axis:
            return Self.backward[ShapeUG, ShapeT](ug, t, axis.value().to_int())
        else:
            return Self.backward[ShapeUG, ShapeT](ug, t)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of mean.
        """
        # d(mean(t)) / dt = 1 / t.num_elements()
        var res_grad = Tensor[dtype](ShapeT)

        var grad: SIMD[dtype, 1] = 1.0 / ShapeT.num_elements()

        grad = (
            grad * ug[0]
        )  # because ug is a tensor of size 1 when mean is used without an axis

        @parameter
        fn v_mean_d[nelts: Int](i: Int):
            res_grad.store[nelts](i, grad)

        vectorize[v_mean_d, nelts](ShapeT.num_elements())

        return res_grad ^

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
        """
        Backward operation of mean.
        """
        # d(mean(t)) / dt = 1 / t.dim(axis)
        var res_grad = Tensor[dtype](ShapeT)

        var grad: SIMD[dtype, 1] = 1.0 / ShapeT[axis]

        fill(res_grad, grad)

        elwise_op[ShapeT, ShapeUG, mul](res_grad, res_grad, ug)

        return res_grad ^


@register_passable("trivial")
struct Max:
    @staticmethod
    fn result_shape(t_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var axis = attributes["axis"]

        if axis:
            return get_reduce_shape(t_shape, axis.value().to_int())
        else:
            return TensorShape(1)

    @staticmethod
    fn forward[
        ShapeT: TensorShape, Attributes: AttributeVector
    ](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the max operation.
        """

        alias axis = Attributes["axis"]

        @parameter
        if axis:
            tmax(res, t, axis.value().to_int())
        else:
            res[0] = tmax(t)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape, Attributes: AttributeVector
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of max.
        """
        alias axis = Attributes["axis"]

        @parameter
        if axis:
            return Self.backward[ShapeUG, ShapeT](ug, t, axis.value().to_int())
        else:
            return Self.backward[ShapeUG, ShapeT](ug, t)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of max.
        """
        # This could be changed to something like in tinygrad:
        # max_1s = CMPEQ(original_tensor, expanded(max_tensor), axis=axis)
        # sum_max_1s = SUM(max_1s)
        # div_sum_max_1s = DIV(max_1, sum_max_1s)

        # The selected element gradient is 1.0, the others are 0.0. And if there are
        # multiple max values, the gradient is divided by the number of max
        # values (1/n) for each max value.

        var res_grad = Tensor[dtype](ShapeT)

        # ug_shape size is 1
        var max_res = tmax(t)
        var sum_eq: SIMD[dtype, 1] = 0
        for i in range(t.num_elements()):
            if t[i] == max_res:
                sum_eq += 1

        var factor = 1 / sum_eq
        for i in range(res_grad.num_elements()):
            if t[i] == max_res:
                res_grad[i] = factor * ug[0]

        return res_grad ^

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype], axis: Int) -> Tensor[dtype]:
        """
        Backward operation of max.
        """
        # The selected element gradient is 1.0, the others are 0.0. And if there are
        # multiple max values, the gradient is divided by the number of max
        # values (1/n) for each max value.

        var res_grad = Tensor[dtype](ShapeT)
        var max_res = Tensor[dtype](ShapeUG)
        alias strides = ShapeT.strides()

        tmax(
            max_res, t, axis
        )  # To not calculate this again we could receive the result of the forward pass as a parameter

        for i in range(max_res.num_elements()):
            var index_base = (i % strides[axis]) + (i // strides[axis]) * (
                strides[axis] * t.dim(axis)
            )

            var count_1s: SIMD[dtype, 1] = 0
            # Count the number of values equal to max_res
            for j in range(t.dim(axis)):
                var index = index_base + j * strides[axis]
                if t[index] == max_res[i]:
                    count_1s += 1
            # Divide 1.0 by the number of max values (n) and multiply by upper gradient value
            var factor = 1 / count_1s
            for j in range(t.dim(axis)):
                var index = index_base + j * strides[axis]
                if t[index] == max_res[i]:
                    res_grad[index] = factor * ug[i]

        return res_grad ^


@register_passable("trivial")
struct Transpose:
    @staticmethod
    fn result_shape(t_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var axes = attributes["axes"]  # axes to be permuted

        var rank = t_shape.rank()
        var shape = StaticIntTuple[max_rank]()

        if axes:
            # NOTE: axis has to be the size of rank of the tensor
            var axes_shape = axes.value().to_shape()
            for i in range(rank):
                shape[i] = t_shape[axes_shape[i]]
        else:
            for i in range(rank):
                shape[i] = t_shape[rank - i - 1]

        return TensorShape(rank=rank, shape=shape)

    @staticmethod
    fn forward[
        ShapeT: TensorShape, Attributes: AttributeVector
    ](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the transpose operation.
        """
        alias axes = Attributes["axes"]

        @parameter
        if axes:
            var axes_shape = axes.value().to_shape()
            transpose(res, t, axes_shape)
        else:

            fn create_transpose_axes() -> TensorShape:
                var rank = ShapeT.rank()
                var axes = StaticIntTuple[max_rank]()
                for i in range(rank):
                    axes[i] = rank - i - 1
                return TensorShape(rank=rank, shape=axes)

            alias axes_shape = create_transpose_axes()

            transpose(res, t, axes_shape)

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape, Attributes: AttributeVector
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of transpose.
        """
        # No local gradient. Transpose is its own inverse.
        alias axes = Attributes["axes"]

        var res_grad = Tensor[dtype](ShapeT)

        @parameter
        if axes:

            fn create_inverse_axes() -> TensorShape:
                var axes_shape = axes.value().to_shape()

                var rank = axes_shape.rank()
                var axes_shape_inv = StaticIntTuple[max_rank]()

                for i in range(rank):
                    axes_shape_inv[axes_shape[i]] = i

                return TensorShape(rank=rank, shape=axes_shape_inv)

            alias axes_shape_inv = create_inverse_axes()

            transpose(res_grad, ug, axes_shape_inv)
        else:

            fn create_transpose_axes() -> TensorShape:
                var rank = ShapeT.rank()
                var axes = StaticIntTuple[max_rank]()
                for i in range(rank):
                    axes[i] = rank - i - 1
                return TensorShape(axes)

            alias axes_shape_inv = create_transpose_axes()

            transpose(res_grad, ug, axes_shape_inv)

        return res_grad ^


@register_passable("trivial")
struct Flatten:
    @staticmethod
    fn result_shape(t_shape: TensorShape) -> TensorShape:
        return TensorShape(t_shape.num_elements())

    @staticmethod
    fn forward[ShapeT: TensorShape](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the flatten operation.
        """
        memcpy(res.data(), t.data(), ShapeT.num_elements())

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of flatten.
        """
        var res_grad = Tensor[dtype](ShapeT)
        memcpy(res_grad.data(), ug.data(), ShapeUG.num_elements())

        return res_grad ^


@register_passable("trivial")
struct Reshape:
    @staticmethod
    fn result_shape(t_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var new_shape = attributes["shape"]
        return new_shape.value().to_shape()

    @staticmethod
    fn forward[ShapeT: TensorShape](inout res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the reshape operation.
        """
        memcpy(res.data(), t.data(), ShapeT.num_elements())

    @staticmethod
    fn backward[
        ShapeUG: TensorShape, ShapeT: TensorShape
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """
        Backward operation of reshape.
        """
        var res_grad = Tensor[dtype](ShapeT)
        memcpy(res_grad.data(), ug.data(), ShapeUG.num_elements())

        return res_grad ^
