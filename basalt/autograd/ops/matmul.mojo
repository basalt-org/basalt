from basalt.utils.tensorutils import transpose_2D
from algorithm import vectorize, parallelize


@always_inline
fn calculate_block[
    M: Int, N: Int, K: Int, BLOCK_M: Int, BLOCK_N: Int, nelts: Int
](
    res: DTypePointer[dtype],
    t1: DTypePointer[dtype],
    t2: DTypePointer[dtype],
    bm: Int,
    bn: Int,
):
    # Compute tile
    var acc = stack_allocation[BLOCK_M * BLOCK_N, dtype]()
    memset_zero[dtype](acc, BLOCK_M * BLOCK_N)

    for k in range(K):

        @unroll
        for m in range(BLOCK_M):

            @parameter
            fn inner_n[nelts: Int](n: Int):
                acc.store[width=nelts](
                    m * BLOCK_N + n,
                    SIMD[dtype, nelts]
                    .splat(t1[(bm + m) * K + k])
                    .fma(
                        t2.load[width=nelts](k * N + (bn + n)),
                        acc.load[width=nelts](m * BLOCK_N + n),
                    ),
                )

            vectorize[inner_n, nelts](BLOCK_N)

    # Store tile
    for m in range(BLOCK_M):

        @parameter
        fn vec_store[nelts: Int](n: Int):
            res.store[width=nelts](
                (bm + m) * N + (bn + n), acc.load[width=nelts](m * BLOCK_N + n)
            )

        vectorize[vec_store, nelts](BLOCK_N)


@parameter
@always_inline
fn dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](inout res: Tensor[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
    dot[t1_shape, t2_shape](res.data(), t1.data(), t2.data())


@parameter
@always_inline
fn dot[
    t1_shape: TensorShape, t2_shape: TensorShape
](res: DTypePointer[dtype], t1: DTypePointer[dtype], t2: DTypePointer[dtype]):
    alias M = t1_shape[0]  # t1[0]
    alias K = t1_shape[1]  # t1[1], t2[0]
    alias N = t2_shape[1]  # t2[1]

    # simdwidthof[dtype]() = 8 for float32
    alias nelts = simdwidthof[dtype]()
    alias BLOCK_N = 8 * 2
    alias BLOCK_M = 6
    alias THREADS = 6  # num_logical_cores()

    alias BLOCK_N_REMAINDER = N % BLOCK_N
    alias BLOCK_M_REMAINDER = M % BLOCK_M

    @parameter
    fn bm_par(m_outer: Int):
        var bm = m_outer * BLOCK_M

        for n_outer in range(0, N // BLOCK_N):
            var bn = n_outer * BLOCK_N

            calculate_block[M, N, K, BLOCK_M, BLOCK_N, nelts](res, t1, t2, bm, bn)

        # Handle the remainder of N
        @parameter
        if BLOCK_N_REMAINDER > 0:
            var bn = N - BLOCK_N_REMAINDER

            calculate_block[M, N, K, BLOCK_M, BLOCK_N_REMAINDER, nelts](
                res, t1, t2, bm, bn
            )

    parallelize[bm_par](M // BLOCK_M, M // BLOCK_M)

    # Handle the remainder of M
    @parameter
    if BLOCK_M_REMAINDER > 0:
        var bm = M - BLOCK_M_REMAINDER

        for n_outer in range(0, N // BLOCK_N):
            var bn = n_outer * BLOCK_N

            calculate_block[M, N, K, BLOCK_M_REMAINDER, BLOCK_N, nelts](
                res, t1, t2, bm, bn
            )

        # Handle corner remainder
        @parameter
        if BLOCK_N_REMAINDER > 0:
            var bn = N - BLOCK_N_REMAINDER

            calculate_block[M, N, K, BLOCK_M_REMAINDER, BLOCK_N_REMAINDER, nelts](
                res, t1, t2, bm, bn
            )


fn dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](inout C: DTypePointer[dtype], A: DTypePointer[dtype], B: DTypePointer[dtype]):
    dot[A_shape, TensorShape(B_shape[1], B_shape[0])](C, A, transpose_2D[B_shape](B))


fn dot_transpose_t2[
    A_shape: TensorShape, B_shape: TensorShape
](inout C: Tensor[dtype], A: Tensor[dtype], B: Tensor[dtype]):
    memset_zero[dtype](C.data(), C.num_elements())

    dot[A_shape, TensorShape(B_shape[1], B_shape[0])](C, A, transpose_2D[B_shape](B))

    # @parameter
    # fn calc_row(i: Int):
    #     for j in range(B_shape[0]):

    #         @parameter
    #         fn calc_row_A_B[nelts: Int](k: Int):
    #             var A_pos = i * A.dim(1) + k
    #             var B_pos = j * A.dim(1) + k
    #             var t_new_pos = i * C.dim(1) + j

    #             C[t_new_pos] += (
    #                 A.load[nelts](A_pos) * B.load[nelts](B_pos)
    #             ).reduce_add()

    #         vectorize[calc_row_A_B, nelts, size=A_shape[1]]()

    # parallelize[calc_row](A_shape[0], 1)


fn dot_transpose_t1[
    A_shape: TensorShape, B_shape: TensorShape
](inout C: Tensor[dtype], A: Tensor[dtype], B: Tensor[dtype]):
    memset_zero[dtype](C.data(), C.num_elements())

    dot[TensorShape(A_shape[1], A_shape[0]), B_shape](C, transpose_2D[A_shape](A), B)

    # @parameter
    # fn calc_row(i: Int):
    #     for j in range(A_shape[0]):

    #         @parameter
    #         fn calc_row_t_new_B[nelts: Int](k: Int):
    #             var A_pos = j * A.dim(1) + i
    #             var B_pos = j * B.dim(1) + k
    #             var t_new_pos = i * C.dim(1) + k

    #             C.store[nelts](
    #                 t_new_pos,
    #                 C.load[nelts](t_new_pos)
    #                 + A[A_pos] * B.load[nelts](B_pos),
    #             )

    #         vectorize[calc_row_t_new_B, nelts, size=B_shape[1]]()

    # parallelize[calc_row](A_shape[1], 1)
