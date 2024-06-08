@always_inline
fn add[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a + b


@always_inline
fn sub[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a - b


@always_inline
fn mul[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a * b


@always_inline
fn div[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
    dtype, simd_width
]:
    return a / b


@always_inline
fn round_simd[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    return round(x)
