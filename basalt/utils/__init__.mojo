from memory.unsafe import bitcast


@always_inline("nodebug")
fn q_sqrt(value: Float32) -> Float32:
    var y = bitcast[DType.float32](0x5F3759DF - (bitcast[DType.uint32](value) >> 1))
    return y * (1.5 - 0.5 * value * y * y)
