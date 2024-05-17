from math import min, abs, max

from basalt import dtype
from basalt import Graph, Symbol, OP
from basalt import Tensor, TensorShape
from basalt.autograd.attributes import AttributeVector, Attribute
from basalt.utils.itertools import product


fn _scale_indeces(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> List[Scalar[dtype]]:    
    var M = int(scale * N)
    var indeces = List[Scalar[dtype]]()
    indeces.reserve(M)

    if align_corners:
        for i in range(M):
            indeces.append(i * ((N - 1) / (M - 1)))
    else:
        var step = 1 / scale
        var start = ((M - 1) * step - N + 1) / 2
        for i in range(M):
            indeces.append(i * step - start)

    return indeces ^


fn nearest_coeffs(N: Int, scale: Scalar[dtype], dim: Int, ndims: Int) -> List[Int]:
    
    @parameter
    fn round_to_index(number: Scalar[dtype]) -> Int:
        return int(number + 0.5) if number > 0 else int(number - 0.5)
    
    var indeces = List[Int]()
    var scaled = _scale_indeces(N, scale, True, dim, ndims)
    var scaled_len = len(scaled)
    indeces.reserve(scaled_len)
    
    for i in range(scaled_len):
        indeces.append(round_to_index(scaled[i]))
    return indeces ^


fn linear_coeffs(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> Tuple[List[Int], List[Int]]:
    @parameter
    fn compute_source_index(dst_index: Int) -> Scalar[dtype]:
        if align_corners:
            return dst_index / scale
        return (dst_index + 0.5) / (scale - 0.5)

    var indices = List[Int]()
    var weights = List[Int]()
    indices.reserve(2 * N)
    weights.reserve(2 * N)

    for i in range(N):
        var src_index = compute_source_index(i)
        var lower_index = int(src_index)
        var upper_index = min(lower_index + 1, N - 1)
        var lower_weight = int(upper_index - src_index) # NOTE: Shouldn't this be float?
        var upper_weight = 1 - lower_weight # NOTE: Shouldn't this be float?

        indices.append(lower_index)
        indices.append(upper_index)
        weights.append(lower_weight)
        weights.append(upper_weight)

    return indices, weights

fn cubic_coeffs(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> Tuple[List[Int], List[Int]]:
    @parameter
    fn compute_source_index(dst_index: Int) -> Scalar[dtype]:
        if align_corners:
            return dst_index / scale
        return (dst_index + 0.5) / (scale - 0.5)
    
    @parameter
    fn cubic_weight(x: Scalar[dtype]) -> Scalar[dtype]:
        var abs_x = abs(x)

        if abs_x <= 1:
            return (1.5 * abs_x - 2.5) * abs_x * abs_x + 1
        elif abs_x <= 2:
            return ((-0.5 * abs_x + 2.5) * abs_x - 4) * abs_x + 2

        return 0

    var indices = List[Int]()
    var weights = List[Int]()
    indices.reserve(N * 4)
    weights.reserve(N * 4)

    for i in range(N):
        var src_index = compute_source_index(i)
        var src_index_floor = int(src_index)
        
        for j in range(-1, 3):
            var index = min(max(src_index_floor + j, 0), N - 1)
            var weight = int(cubic_weight(src_index - index)) # NOTE: Shouldn't this be float?
            indices.append(index)
            weights.append(weight)

    return indices, weights


fn interpolate_nd[
    indices_fn: fn (Int, Scalar[dtype], Bool, Int, Int) -> Tuple[List[Int], List[Int]],
](inout g: Graph, input: Symbol, scale_factors: List[Scalar[dtype]], align_corners: Bool) -> Symbol:

    var spatial_dims = input.shape.rank() - 2
    
    var indeces_weights = List[Tuple[List[Int], List[Int]]]()
    for i in range(spatial_dims):
        indeces_weights.append(
            indices_fn(
                input.shape[i + 2],
                scale_factors[i],
                align_corners,
                i,
                spatial_dims,
            )
        )

    # TODO: interpolation logic
    # for idx_weight in product(indeces_weights):
    #     ...

    return Symbol(-1, dtype, TensorShape(), False)


fn Upsample(
    inout g: Graph,
    input: Symbol,
    mode: StringLiteral,
    scale_factors: List[Scalar[dtype]],
    align_corners: Bool = False,
) -> Symbol:

    # Assumption: A scale needs to be provided for each spatial dimension.
    # input shape (B, C, *N) with batch and channel considered as non-spatial dimensions.
    # input.shape.rank() - 2 == len(scale_factor)
    var spatial_dims = input.shape.rank() - 2

    var res: Symbol
    var attributes = AttributeVector()
    var INDEX_LITERALS = List[StringLiteral]("dim_2i", "dim_3i", "dim_4i")

    if mode == "nearest":
        # Nearest neighbor interpolation --> input[:, :, *indeces]
        for i in range(spatial_dims):            
            attributes.append(
                Attribute(
                    INDEX_LITERALS[i],
                    nearest_coeffs(input.shape[i + 2], scale_factors[i], i, spatial_dims)
                )
            )

        res = g.op(OP.INDEX, input, attributes=attributes)

    elif mode == "linear":
        res = interpolate_nd[linear_coeffs](g, 
            input,
            scale_factors,
            align_corners
        )
    
    elif mode == "cubic":
        res = interpolate_nd[cubic_coeffs](g, 
            input,
            scale_factors,
            align_corners
        )
    else:
        res = input

    return res

