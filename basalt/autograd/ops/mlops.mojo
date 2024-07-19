from algorithm import vectorize, parallelize
from math import exp, floor, ceil
from utils.numerics import min_finite, max_finite
from utils.static_tuple import StaticTuple

from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import elwise_transform
from basalt.utils.itertools import product
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

        return res_grad^


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

        return res_grad^


struct LEAKYRELU:
    @staticmethod
    fn result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of leaky_relu."""

        fn leaky_relu[
            type: DType,
            simd_width: Int,
        ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
            var negative_slope = attributes["negative_slope"].value().to_scalar[
                type
            ]()
            return (x > 0).select(x, x * negative_slope)

        elwise_transform[leaky_relu](res, t1)

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of leaky_relu."""

        @always_inline
        fn leaky_relu_bw[
            type: DType, simd_width: Int
        ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
            var negative_slope = attributes["negative_slope"].value().to_scalar[
                type
            ]()

            return (x > 0).select[type](1, negative_slope)

        var res_grad = Tensor[dtype](ug_shape)

        @parameter
        fn vec_leaky_relu_bw[nelts: Int](idx: Int):
            res_grad.store[nelts](
                idx,
                leaky_relu_bw(t1.load[nelts](idx)) * ug.load[nelts](idx),
            )

        vectorize[vec_leaky_relu_bw, nelts](ug_shape.num_elements())

        return res_grad^


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

        return res_grad^


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

        var min_val = min_attr.value().to_scalar[
            dtype
        ]() if min_attr else min_finite[dtype]()
        var max_val = max_attr.value().to_scalar[
            dtype
        ]() if max_attr else max_finite[dtype]()

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

        var min_val = min_attr.value().to_scalar[
            dtype
        ]() if min_attr else min_finite[dtype]()
        var max_val = max_attr.value().to_scalar[
            dtype
        ]() if max_attr else max_finite[dtype]()

        var res_grad = Tensor[dtype](t_shape)

        @parameter
        fn vec_clip_bw[nelts: Int](i: Int):
            var val = t.load[nelts](i)
            res_grad.store[nelts](
                i,
                ((val >= min_val) * (val <= max_val)).select(
                    ug.load[nelts](i), 0
                ),
            )

        vectorize[vec_clip_bw, nelts, size = t_shape.num_elements()]()

        return res_grad^


struct SQUEEZE:
    @staticmethod
    fn result_shape(
        t1_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
        var dim = attributes["dims"]
        var dims_to_squeeze = dim.value().to_shape() if dim else TensorShape()

        var new_shape = List[Int]()
        for i in range(t1_shape.rank()):
            if (not dim and t1_shape[i] == 1) or (
                i in dims_to_squeeze and t1_shape[i] == 1
            ):
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
        return res_grad^


struct UNSQUEEZE:
    @staticmethod
    fn result_shape(
        t1_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
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
        return res_grad^


struct SLICE:
    @staticmethod
    fn adjust_boundary(slice: Int, dim_size: Int) -> Int:
        # Adjust negative indices & ensure they are within bounds.
        var s = slice if slice >= 0 else dim_size + slice
        return max(min(s, dim_size), 0)

    @staticmethod
    fn default_starts(shape: TensorShape) -> List[Int]:
        var starts = List[Int]()
        for i in range(shape.rank()):
            starts.append(0)
        return starts^

    @staticmethod
    fn default_ends(shape: TensorShape) -> List[Int]:
        var ends = List[Int]()
        for i in range(shape.rank()):
            ends.append(shape[i])
        return ends^

    @staticmethod
    fn default_steps(shape: TensorShape) -> List[Int]:
        var steps = List[Int]()
        for i in range(shape.rank()):
            steps.append(1)
        return steps^

    @staticmethod
    fn default_axes(shape: TensorShape) -> List[Int]:
        # NOTE: axes can't be negative
        var axes = List[Int]()
        for i in range(shape.rank()):
            axes.append(i)
        return axes^

    @staticmethod
    fn result_shape(
        t1_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
        # NOTE: Starts and ends have to be of the same size
        # NOTE: If axes not provided, starts and ends have to be of the same size as t1_shape
        var starts = attributes["starts"].value().to_shape()
        var ends = attributes["ends"].value().to_shape()
        var steps = attributes["steps"].value().to_shape() if attributes[
            "steps"
        ] else Self.default_steps(starts)
        var axes = attributes["axes"].value().to_shape() if attributes[
            "axes"
        ] else Self.default_axes(t1_shape)

        var new_shape = t1_shape
        for i in range(starts.rank()):
            var axis = axes[i]
            new_shape[axis] = len(
                range(
                    start=Self.adjust_boundary(starts[i], t1_shape[axis]),
                    end=Self.adjust_boundary(ends[i], t1_shape[axis]),
                    step=steps[i],
                )
            )

        return new_shape

    @staticmethod
    fn reorder_positions[
        id: Int
    ](original: TensorShape, axes: TensorShape, t1_shape: TensorShape) -> List[
        Int
    ]:
        # Reorder the starts (id=0), ends (id=1) or steps (id=2) to match the order of the axes
        var updated: List[Int]

        @parameter
        if id == 0:
            updated = Self.default_starts(t1_shape)
        elif id == 1:
            updated = Self.default_ends(t1_shape)
        else:
            updated = Self.default_steps(t1_shape)

        for i in range(axes.rank()):
            var axis = axes[i]
            updated[axis] = original[i] if id == 2 else Self.adjust_boundary(
                original[i], t1_shape[axis]
            )

        return updated^

    # NOTE: For now you can't have recursive function as parameter functions.
    # NOTE: From testing it seems a recursive function is almost the same speed as doing multiple nested for loops.
    @staticmethod
    fn recursive_iters_slice[
        shape: TensorShape,
        original_shape: TensorShape,
        steps: List[Int],
        starts: List[Int],
        ends: List[Int],
        backward_op: Bool = False,
    ](
        inout res: Tensor[dtype],
        t1: Tensor[dtype],
        last_dims: Int,
        position: Int,
        last_position: Int,
        idx: Int,
        idx_original: Int,
    ):
        alias strides = shape.strides()
        alias t1_strides = original_shape.strides()

        var idx_temp = idx
        var idx_original_temp = starts[position] * t1_strides[
            position
        ] + idx_original

        if position == last_position + 1:
            # Work on the last dimensions
            alias position = shape.rank() - 1
            alias stride = t1_strides[position] * steps[position]

            @parameter
            fn v_slice[nelts: Int](k: Int):
                @parameter
                if not backward_op:

                    @parameter
                    if steps[position] == 1:
                        res.store[nelts](
                            idx_temp + k, t1.load[nelts](idx_original_temp)
                        )
                    else:
                        res.store[nelts](
                            idx_temp + k,
                            t1.data()
                            .offset(idx_original_temp)
                            .simd_strided_load[nelts](stride),
                        )
                else:

                    @parameter
                    if steps[position] == 1:
                        res.store[nelts](idx_original_temp, t1.load[nelts](idx_temp + k))
                    else:
                        res.data().offset(idx_original_temp).simd_strided_store[width=nelts](
                            t1.load[nelts](idx_temp + k),
                            stride
                        )

                idx_original_temp += stride * nelts

            vectorize[v_slice, nelts](last_dims)

            return

        for _ in range(shape[position]):
            Self.recursive_iters_slice[
                shape, original_shape, steps, starts, ends, backward_op
            ](
                res,
                t1,
                last_dims,
                position + 1,
                last_position,
                idx_temp,
                idx_original_temp,
            )

            idx_temp += strides[position]
            idx_original_temp += steps[position] * t1_strides[position]

    @staticmethod
    fn slice_kernel[
        res_shape: TensorShape,
        original_shape: TensorShape,
        steps: List[Int],
        starts: List[Int],
        ends: List[Int],
        backward_op: Bool = False,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        alias strides = original_shape.strides()

        # Get the dimensions for vectorization
        var last_dims = 1
        var positions_to_skip = 0
        for i in range(res_shape.rank() - 1, -1, -1):
            if steps[i] != 1 and i != res_shape.rank() - 1:
                break
            last_dims *= res_shape[i]
            positions_to_skip += 1
            if starts[i] != 0 or ends[i] != original_shape[i] or steps[i] != 1:
                break

        # Get the dimensions for the first loop
        var first_dims = 1
        var start_position = 0
        for i in range(res_shape.rank() - positions_to_skip):
            if steps[i] != 1 or starts[i] != 0 or ends[i] != original_shape[i]:
                break
            first_dims *= res_shape[i]
            start_position += 1

        var middle_dims = res_shape.num_elements() // last_dims // first_dims

        @parameter
        fn p_slice(i: Int):
            Self.recursive_iters_slice[
                res_shape, original_shape, steps, starts, ends, backward_op
            ](
                res,
                t1,
                last_dims,
                start_position,
                res_shape.rank() - 1 - positions_to_skip,
                i * middle_dims * last_dims,
                i * strides[start_position - 1],
            )

        parallelize[p_slice](first_dims)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        alias axes = attributes["axes"].value().to_shape() if attributes[
            "axes"
        ] else Self.default_axes(t1_shape)
        alias starts = Self.reorder_positions[0](
            attributes["starts"].value().to_shape(), axes, t1_shape
        )
        alias ends = Self.reorder_positions[1](
            attributes["ends"].value().to_shape(), axes, t1_shape
        )
        alias steps = Self.reorder_positions[2](
            attributes["steps"].value().to_shape(), axes, t1_shape
        ) if attributes["steps"] else Self.default_steps(t1_shape)

        alias res_shape = Self.result_shape(t1_shape, attributes)

        Self.slice_kernel[res_shape, t1_shape, steps, starts, ends, False](
            res, t1
        )

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        attributes: AttributeVector = AttributeVector(),
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        alias axes = attributes["axes"].value().to_shape() if attributes[
            "axes"
        ] else Self.default_axes(t1_shape)
        alias starts = Self.reorder_positions[0](
            attributes["starts"].value().to_shape(), axes, t1_shape
        )
        alias ends = Self.reorder_positions[1](
            attributes["ends"].value().to_shape(), axes, t1_shape
        )
        alias steps = Self.reorder_positions[2](
            attributes["steps"].value().to_shape(), axes, t1_shape
        ) if attributes["steps"] else Self.default_steps(t1_shape)

        var res_grad = Tensor[dtype](t1_shape)
        
        Self.slice_kernel[ug_shape, t1_shape, steps, starts, ends, True](res_grad, ug)
        
        return res_grad ^


struct INDEX:
    @staticmethod
    fn adjust_boundary(slice: Int, dim_size: Int) -> Int:
        # Adjust negative indices & ensure they are within bounds.
        var s = slice if slice >= 0 else dim_size + slice
        return max(min(s, dim_size), 0)

    @staticmethod
    fn to_indeces(shape: TensorShape, attrs: AttributeVector) -> List[List[Int]]:
        var SLICE_LITERALS = List[StringLiteral]("dim_0s", "dim_1s", "dim_2s", "dim_3s", "dim_4s", "dim_5s", "dim_6s", "dim_7s")
        var INDEX_LITERALS = List[StringLiteral]("dim_0i", "dim_1i", "dim_2i", "dim_3i", "dim_4i", "dim_5i", "dim_6i", "dim_7i")

        var indeces = List[List[Int]]()
        for dim in range(shape.rank()):
            var temp = List[Int]()
            
            # Option 1: Slice
            if attrs[SLICE_LITERALS[dim]]:
                var slice = attrs[SLICE_LITERALS[dim]].value().to_shape()
                var step = slice[2] if slice.rank() == 3 else 1
                for i in range(
                    start=Self.adjust_boundary(slice[0], shape[dim]),
                    end=Self.adjust_boundary(slice[1], shape[dim]),
                    step=step
                ):
                    temp.append(i)

            # Option 2: Indeces
            elif attrs[INDEX_LITERALS[dim]]:
                var indeces = attrs[INDEX_LITERALS[dim]].value().to_shape()
                for i in range(indeces.rank()):
                    temp.append(indeces[i])

            # All indeces
            else:
                for i in range(shape[dim]):
                    temp.append(i)

            indeces.append(temp)
        
        return indeces ^

    @staticmethod
    fn result_shape(shape: TensorShape, attrs: AttributeVector) -> TensorShape:
        var indeces = Self.to_indeces(shape, attrs)
        var new_shape = List[Int]()
        for i in range(shape.rank()):
            new_shape.append(len(indeces[i]))
        return TensorShape(new_shape)

    @staticmethod
    fn map_indeces[
        nelts: Int,
        strides: TensorShape,
        indeces: List[List[Int]],
    ](idx: Int) -> SIMD[DType.int64, nelts]:
        alias indeces_product = product(indeces)

        var temp = SIMD[DType.int64, nelts]()
        for i in range(idx, idx + nelts):
            var comb = indeces_product[i]
            var flat_index = 0

            for dim in range(len(comb)):
                flat_index += comb[dim] * strides[dim]

            temp[i % nelts] = flat_index

        return temp

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        alias indeces = Self.to_indeces(t1_shape, attributes)
        alias strides = t1_shape.strides()
        alias total_length = len(product(indeces))

        @parameter
        fn vec_index[nelts: Int](i: Int):

            res.store[nelts](i,
                t1.data().gather(Self.map_indeces[nelts, strides, indeces](i))
            )

        vectorize[vec_index, nelts](total_length)


    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        attributes: AttributeVector = AttributeVector(),
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        alias indeces = Self.to_indeces(t1_shape, attributes)
        alias strides = t1_shape.strides()
        alias total_length = len(product(indeces))

        var res_grad = Tensor[dtype](t1_shape)

        @parameter
        fn vec_index[nelts: Int](i: Int):

            var offset = Self.map_indeces[nelts, strides, indeces](i)
            
            # res_grad.data().scatter(
            #     offset,
            #     res_grad.data().gather(offset) + ug.load[nelts](i),
            # )
            # BUG: Edge case in vectorization:
            # When the offset = [0, 2, 4, 0] and ug = [1, 1, 1, 1]
            # It doesn't scatter to index 0 twice as it should be: res_grad[0] += 1 + 1
            
            # Workaround
            var u = ug.load[nelts](i)
            for j in range(nelts):
                res_grad[int(offset[j])] += u[j]

        vectorize[vec_index, nelts](total_length)

        return res_grad^


struct UPSAMPLE:
    @staticmethod
    fn result_shape(t1_shape: TensorShape, attributes: AttributeVector) -> TensorShape:
        var scales = attributes["scales"].value().to_shape()
        var mode = attributes["mode"].value().to_string()

        var new_shape = List[Int]()
        for i in range(0, t1_shape.rank()):
            if i < 2:
                new_shape.append(t1_shape[i])
            else:
                new_shape.append(t1_shape[i] * scales[i - 2])

        return TensorShape(new_shape)

    @staticmethod
    fn recursive_iter[pos_shape: Int, shape: TensorShape, scales: TensorShape](inout res: Tensor[dtype], t1: Tensor[dtype], strides_res: StaticIntTuple[8], index_t1: Int, index_res: Int):
        alias end_pos = shape.rank() - 1
        alias strides = shape.strides()

        @parameter
        if pos_shape >= end_pos:
            @parameter
            fn v_iter[nelts: Int](i: Int):
                var values = t1.load[nelts](index_t1 + i)

                var offset_res = index_res + i * scales[end_pos - 2]
                for j in range(nelts * scales[pos_shape - 2]):
                    var temp = j // scales[pos_shape - 2]

                    res[offset_res + j] = values[temp]

            vectorize[v_iter, nelts](shape[pos_shape])
            
            return
        else:
            for i in range(shape[pos_shape] * scales[pos_shape - 2]):
                var temp_i = i // scales[pos_shape - 2]
                var temp_index_t1 = temp_i * strides[pos_shape] + index_t1
                var temp_index_res = i * strides_res[pos_shape] + index_res

                Self.recursive_iter[pos_shape + 1, shape, scales](res, t1, strides_res, temp_index_t1, temp_index_res)

    @staticmethod
    fn forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](inout res: Tensor[dtype], t1: Tensor[dtype]):
        # Input is [N, C, D in, H in, W in], N is batch size and C is number of channels. Ranks 3-D, 4-D or 5-D tensors (only works on the spatial dimensions).
        alias scales = attributes["scales"].value().to_shape() # Has to match spatial input dims (the last dimensions D, H and W)
        alias mode = attributes["mode"].value().to_string()
        # alias align_corners = attributes["align_corners"].value().to_bool() if attributes["align_corners"] else false

        @parameter
        fn get_coordination_mode() -> String:
            if mode == "linear" or mode == "bilinear":
                return "half_pixel"
            else:
                return "asymmetric"
        alias coordination_transforamtion = get_coordination_mode()

        alias strides = t1_shape.strides()
        var strides_res = res.strides()
        
        var res_shape = res.shape()

        alias first_loop = t1_shape[0] * t1_shape[1]

        @always_inline
        fn pos_asymmetric(pos: Int, scale: Int) -> Int:
            return pos // scale
        
        @always_inline
        fn pos_half_pixel(pos: Int, scale: Int) -> Float64:
            return max(0.0, (pos + 0.5) / scale - 0.5)


        @parameter
        @always_inline
        fn get_value_interpolate[size: Int](
            indeces_t1: StaticTuple[Float64, size]
        ) -> SIMD[t1.dtype, 1]:
            @parameter
            if mode == "nearest":
                var indeces_t1_sum = indeces_t1[0]
                @parameter
                for i in range(1, size):
                    indeces_t1_sum += indeces_t1[i] * strides[i + 1]

                return t1[int(indeces_t1_sum)]
            elif mode == "linear":
                var t1_pos_floor = floor(indeces_t1[1])
                var t1_pos_ceil = min(ceil(indeces_t1[1]), t1_shape[2] - 1)

                var v1 = t1[int(indeces_t1[0]) + int(t1_pos_floor)]
                var v2 = t1[int(indeces_t1[0]) + int(t1_pos_ceil)]

                return v1 + (v2 - v1) * (indeces_t1[1] - t1_pos_floor)
            elif mode == "bilinear":
                var t1_pos_floor_y = floor(indeces_t1[1])
                var t1_pos_ceil_y = min(ceil(indeces_t1[1]), t1_shape[2] - 1)

                var t1_pos_floor_x = floor(indeces_t1[2])
                var t1_pos_ceil_x = min(ceil(indeces_t1[2]), t1_shape[3] - 1)

                var v1 = t1[int(indeces_t1[0]) + int(t1_pos_floor_y) * strides[2] + int(t1_pos_floor_x) * strides[3]]
                var v2 = t1[int(indeces_t1[0]) + int(t1_pos_floor_y) * strides[2] + int(t1_pos_ceil_x) * strides[3]]
                var v3 = t1[int(indeces_t1[0]) + int(t1_pos_ceil_y) * strides[2] + int(t1_pos_floor_x) * strides[3]]
                var v4 = t1[int(indeces_t1[0]) + int(t1_pos_ceil_y) * strides[2] + int(t1_pos_ceil_x) * strides[3]]

                var wy = indeces_t1[1] - t1_pos_floor_y
                var wx = indeces_t1[2] - t1_pos_floor_x

                var top_interp = v1 + (v2 - v1) * wx
                var bottom_interp = v3 + (v4 - v3) * wx
    
                return top_interp + (bottom_interp - top_interp) * wy
            else:
                return 0

        @always_inline
        fn get_t1_position(
            pos: Int, scale: Int, dim: Int
        ) -> Float64:
            @parameter
            if coordination_transforamtion == "asymmetric":
                return pos_asymmetric(pos, scale)
            elif coordination_transforamtion == "half_pixel":
                return pos_half_pixel(pos, scale)
            else:
                return 0

        # it is possble to use gather, the only problem is to be able to create a simd arange (vectorized if it is with a for loop it is the same probably). (And from tests it seems to be slower, maybe because i do a lot of casts and because the arange of positions is not vectorized)
        @parameter
        fn p_iter(i: Int):
            var offset_t1 = i * strides[1]
            var offset_res = i * strides_res[1]
    
            @parameter
            if t1_shape.rank() == 3:
                var positions_t1 = StaticTuple[Float64, 2](0)
                var positions_res = StaticIntTuple[2](0)

                positions_res[0] = offset_res
                positions_t1[0] = offset_t1
        
                @parameter
                fn v_iter[nelts: Int](j: Int):
                    positions_res[1] = j

                    var index_res = positions_res[0] + positions_res[1]
                    var values = res.load[nelts](index_res)

                    for k in range(nelts):
                        positions_t1[1] = get_t1_position(j + k, scales[scales.rank() - 1], 0)

                        values[k] = get_value_interpolate(positions_t1)

                    res.store[nelts](index_res, values)

                
                vectorize[v_iter, nelts](res_shape[res.rank() - 1])
            elif t1_shape.rank() == 4:
                var positions_t1 = StaticTuple[Float64, 3](0)
                var positions_res = StaticIntTuple[3](0)

                positions_res[0] = offset_res
                positions_t1[0] = offset_t1

                for j in range(res_shape[2]):
                    positions_res[1] = j * strides_res[2]
                    positions_t1[1] = get_t1_position(j, scales[0], 0)
            
                    @parameter
                    fn v_iter_1[nelts: Int](k: Int):
                        positions_res[2] = k

                        var index_res = positions_res[0] + positions_res[1] + positions_res[2]
                        var values = res.load[nelts](index_res)

                        for l in range(nelts):
                            positions_t1[2] = get_t1_position(k + l, scales[scales.rank() - 1], 1)

                            values[l] = get_value_interpolate(positions_t1)

                        res.store[nelts](index_res, values)
                    
                    vectorize[v_iter_1, nelts](res_shape[res.rank() - 1])

            elif t1_shape.rank() == 5:
                var positions_t1 = StaticTuple[Float64, 4](0)
                var positions_res = StaticIntTuple[4](0)

                positions_res[0] = offset_res
                positions_t1[0] = offset_t1

                for j in range(res.shape()[2]):
                    positions_res[1] = j * strides_res[2]
                    positions_t1[1] = get_t1_position(j, scales[0], 0)
                    for k in range(res.shape()[3]):
                        positions_res[2] = k * strides_res[3]
                        positions_t1[2] = get_t1_position(k, scales[1], 1)
                        
                        @parameter
                        fn v_iter_2[nelts: Int](l: Int):
                            positions_res[3] = l

                            var index_res = positions_res[0] + positions_res[1] + positions_res[2] + positions_res[3]
                            var values = res.load[nelts](index_res)

                            for m in range(nelts):
                                positions_t1[3] = get_t1_position(l + m, scales[scales.rank() - 1], 2)

                                values[m] = get_value_interpolate(positions_t1)

                            res.store[nelts](index_res, values)
                        
                        vectorize[v_iter_2, nelts](res_shape[res.rank() - 1])
            else:
                # Error
                pass    

        parallelize[p_iter](first_loop)

    @staticmethod
    fn backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        attributes: AttributeVector = AttributeVector(),
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        return t1
