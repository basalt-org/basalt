from math import min, max, floor, ceil

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


alias Coeff = Tuple[List[Int], List[Scalar[dtype]]]
alias Coeffs = List[Coeff]

fn linear_coeffs(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> Coeffs:

    var indeces_l = List[Int]()
    var indeces_r = List[Int]()
    var weights_l = List[Scalar[dtype]]()
    var weights_r = List[Scalar[dtype]]()
    for value in _scale_indeces(N, scale, align_corners, dim, ndims):
        var clipped = min[dtype]((max[dtype](value[], 0)), N-1)
        var idx_l = floor(clipped)
        var idx_r = ceil(clipped)

        indeces_l.append(int(idx_l))
        indeces_r.append(int(idx_r))
        weights_l.append(1 - (clipped - idx_l))
        weights_r.append(clipped - idx_l)

    print(len(indeces_l), len(indeces_r), len(weights_l), len(weights_r))

    return List[Coeff](
        Tuple[List[Int]](indeces_l, weights_l),
        Tuple(indeces_r, weights_r),
    )


fn cubic_coeffs(N: Int, scale: Scalar[dtype], align_corners: Bool, dim: Int, ndims: Int) -> Coeffs:
    # TODO
    return List[Coeff](
        Tuple(List[Int](), List[Scalar[dtype]]()),
        Tuple(List[Int](), List[Scalar[dtype]]()),
    )





fn interpolate_nd[
    indices_fn: fn (Int, Scalar[dtype], Bool, Int, Int) -> Coeffs,
](inout g: Graph, input: Symbol, scale_factors: List[Scalar[dtype]], align_corners: Bool) -> Symbol:

    var spatial_dims = input.shape.rank() - 2
    
    var temp = List[Int]()
    var indeces_weights = List[Coeffs]()
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

        temp.append(i)

    @parameter
    fn get_comb_idx(dim: Int, coeff_id: Int) -> List[Int]:
        return indeces_weights[dim][coeff_id].get[0, List[Int]]()

    @parameter
    fn get_comb_weight(dim: Int, coeff_id: Int) -> List[Scalar[dtype]]:
        return indeces_weights[dim][coeff_id].get[1, List[Scalar[dtype]]]()

    var indeces_weights_copy = indeces_weights

    for comb_id in product(List[List[Int]](temp, temp)):
        print("----")

        for i in range(spatial_dims):
            print("D", i,"COMB", comb_id[i])
            print(len(indeces_weights), len(indeces_weights[i]))
            # var temp = indeces_weights[i][comb_id[i]].get[0, List[Int]]()
            var temp = indeces_weights_copy[i][comb_id[i]].get[0, List[Int]]()[0]
            # print(len(temp))
            # var idx = get_comb_idx(i, comb_id[i])
            # var weight = get_comb_weight(i, comb_id[i])

    #             for j in range(len(idx)):
    #                 print(idx[j], weight[j])

        
    # #     for i in range(len(comb_id)):
    # #         var iw_l = indeces_weights[0]
    # #         var iw_r = indeces_weights[1]
            
    # #         print(comb_id[i])


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
        print("[ERROR] Upsampling mode not supported")
        res = input

    return res

