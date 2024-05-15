from basalt import dtype
from basalt import Graph, Symbol, OP
from basalt import Tensor, TensorShape
from basalt.autograd.attributes import AttributeVector, Attribute
from basalt.utils.itertools import product


fn _scale_indeces(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> List[Scalar[dtype]]:    
    var M = int(scale * N)
    var indeces = List[Scalar[dtype]]()
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
    for i in range(len(scaled)):
        indeces.append(round_to_index(scaled[i]))
    return indeces ^


fn linear_coeffs(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> Tuple[List[Int], List[Int]]:
    # TODO
    return (List[Int](), List[Int]())


fn cubic_coeffs(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> Tuple[List[Int], List[Int]]:
    # TODO
    return (List[Int](), List[Int]())


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

    # elif mode == "linear":
    #     res = interpolate_nd[linear_coeffs](g, 
    #         input,
    #         scale_factor,
    #         align_corners
    #     )
    
    # elif mode == "cubic":
    #     res = interpolate_nd[cubic_coeffs](g, 
    #         input,
    #         scale_factor,
    #         align_corners
    #     )
    else:
        res = input

    return res

