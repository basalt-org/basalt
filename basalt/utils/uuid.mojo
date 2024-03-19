
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

    var state: StaticTuple[Self.N, Int32]
    var index: Int

    fn __init__(inout self, seed: Int):
        alias W: Int = 32
        alias F: Int32 = 1812433253
        alias D: Int32 = 0xFFFFFFFF
        
        self.index = Self.N
        self.state = StaticTuple[Self.N, Int32]()
        self.state[0] = seed & D
        
        for i in range(1, Self.N):
            self.state[i] = (F * (self.state[i - 1] ^ (self.state[i - 1] >> (W - 2))) + i) & D
        
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


@register_passable("trivial")
struct UUID(Stringable, CollectionElement):
    var bytes: StaticTuple[16, UInt8]

    fn __init__(inout self):
        self.bytes = StaticTuple[16, UInt8]()

    fn __setitem__(inout self, index: Int, value: UInt8):
        self.bytes[index] = value

    fn __getitem__(self, index: Int) -> UInt8:
        return self.bytes[index]

    fn __eq__(self, other: Self) -> Bool:
        @unroll
        for i in range(16):
            if self.bytes[i] != other.bytes[i]:
                return False
        return True

    fn __str__(self) -> String:
        var result: String = ""
        alias hex_digits: String = "0123456789abcdef"

        @unroll
        for i in range(16):
            if i == 4 or i == 6 or i == 8 or i == 10:
                result += "-"
            result += (
                hex_digits[(self.bytes[i] >> 4).to_int()]
                + hex_digits[(self.bytes[i] & 0xF).to_int()]
            )
        return result


@register_passable("trivial")
struct UUIDGenerator:
    var prng: MersenneTwister

    fn __init__(inout self, seed: Int):
        self.prng = MersenneTwister(seed)

    fn next(inout self) -> UUID:
        var uuid = UUID()

        @unroll
        for i in range(16):
            uuid[i] = self.prng.next_ui8()
        
        # Version 4, variant 10xx
        uuid[6] = 0x40 | (0x0F & uuid[6])
        uuid[8] = 0x80 | (0x3F & uuid[8])
        
        return uuid
