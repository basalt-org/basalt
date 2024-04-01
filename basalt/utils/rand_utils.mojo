from random import rand, randn
from algorithm import vectorize


fn rand_uniform[
    Type: DType
](inout res: Tensor[dtype], low: SIMD[dtype, 1], high: SIMD[dtype, 1]):
    rand[dtype](
        res.data(), res.num_elements()
    )  # Uniform initialize the tensor between 0 and 1

    @parameter
    fn vecscale[nelts: Int](idx: Int):
        res.store[nelts](idx, res.load[nelts](idx) * (high - low) + low)

    vectorize[vecscale, nelts](res.num_elements())


fn rand_normal[dtype: DType](inout res: Tensor[dtype], mean: Float64, std: Float64):
    randn[dtype](
        res.data(), res.num_elements(), mean, std**2
    )  # Normal distribution tensor initialization


@register_passable("trivial")
struct MersenneTwister:
    """
    Pseudo-random generator Mersenne Twister (MT19937-32bit).
    """

    alias N: Int = 624
    alias M: Int = 397
    alias MATRIX_A: Int32 = 0x9908B0DF
    alias UPPER_MASK: Int32 = 0x80000000
    alias LOWER_MASK: Int32 = 0x7FFFFFFF
    alias TEMPERING_MASK_B: Int32 = 0x9D2C5680
    alias TEMPERING_MASK_C: Int32 = 0xEFC60000

    var state: StaticTuple[Int32, Self.N]
    var index: Int

    fn __init__(inout self, seed: Int):
        alias W: Int = 32
        alias F: Int32 = 1812433253
        alias D: Int32 = 0xFFFFFFFF

        self.index = Self.N
        self.state = StaticTuple[
            Int32,
            Self.N,
        ]()
        self.state[0] = seed & D

        for i in range(1, Self.N):
            self.state[i] = (
                F * (self.state[i - 1] ^ (self.state[i - 1] >> (W - 2))) + i
            ) & D

    fn next(inout self) -> Int32:
        if self.index >= Self.N:
            for i in range(Self.N):
                var x = (self.state[i] & Self.UPPER_MASK) + (
                    self.state[(i + 1) % Self.N] & Self.LOWER_MASK
                )
                var xA = x >> 1
                if x % 2 != 0:
                    xA ^= Self.MATRIX_A
                self.state[i] = self.state[(i + Self.M) % Self.N] ^ xA
            self.index = 0

        var y = self.state[self.index]
        y ^= y >> 11
        y ^= (y << 7) & Self.TEMPERING_MASK_B
        y ^= (y << 15) & Self.TEMPERING_MASK_C
        y ^= y >> 18
        self.index += 1

        return y

    fn next_ui8(inout self) -> UInt8:
        return self.next().value & 0xFF
