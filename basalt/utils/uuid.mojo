from math.limit import max_finite
from math import min

@value
@register_passable
struct bytes[
    capacity: Int
](Stringable, CollectionElement):
    """
    Static sequence of bytes.
    """
    var _vector: StaticTuple[capacity, UInt8]

    fn __init__(inout self):
        var _vector = StaticTuple[capacity, UInt8]()
        for i in range(capacity):
            _vector[i] = 0
        self._vector = _vector

    fn __init__(inout self, s: String):
        var _vector = StaticTuple[capacity, UInt8]()
        for i in range(min(len(s), capacity)):
            _vector[i] = ord(s[i])
        self._vector = _vector

    fn __len__(self) -> Int:
        return len(self._vector)

    fn __setitem__(inout self, index: Int, value: UInt8):
        self._vector[index] = value

    fn __getitem__(self, index: Int) -> UInt8:
        return self._vector[index]

    fn __str__(self) -> String:
        var result: String = ""
        for i in range(self.__len__()):
            if self[i].to_int() != 0:
                result += chr(self[i].to_int())
        return result

    fn __eq__(self, other: Self) -> Bool:
        for i in range(self.__len__()):
            if self[i] != other[i]:
                return False
        return True
    
    fn hex(self) -> String:
        var result: String = ""
        alias hex_table: String = "0123456789abcdef"
        for i in range(self.__len__()):
            result += hex_table[((self[i] >> 4) & 0xF).to_int()] + hex_table[(self[i] & 0xF).to_int()]
        return result


# Initialize the generator from a seed
fn mt19937_seed(seed: Int) -> StaticTuple[624, Int32]:
    # Constants for MT19937
    var W: Int32 = 32
    var F: Int32 = 1812433253
    var D: Int32 = 0xFFFFFFFF

    # var MT = StaticIntTuple[624]()
    var MT = StaticTuple[624, Int32]()

    MT[0] = seed
    for i in range(1, 624):
        MT[i] = (F * (MT[i - 1] ^ (MT[i - 1] >> (W - 2))) + i) & D
    
    return MT


fn random_mt(inout MT: StaticTuple[624, Int32], inout index: Int) -> Int32:
    """
    Pseudo-random generator Mersenne Twister (MT19937).
    """

    # Constants for MT19937
    alias N = 624
    alias M = 397
    alias R: Int32 = 31
    alias A: Int32 = 0x9908B0DF
    alias U: Int32 = 11
    alias D: Int32 = 0xFFFFFFFF
    alias S: Int32 = 7
    alias B: Int32 = 0x9D2C5680
    alias T: Int32 = 15
    alias C: Int32 = 0xEFC60000
    alias L: Int32 = 18

    alias LOWER_MASK = (1 << R) - 1
    alias UPPER_MASK = (~LOWER_MASK) & 0xFFFFFFFF

    # Twisting every N numbers
    if index >= N:
        ## Check disregarded for compiler
        # if index > N:
        #     print("[ERROR] Generator was never seeded")      
        for i in range(N):
            var x = (MT[i] & UPPER_MASK) + (MT[(i+1) % N] & LOWER_MASK)
            var xA = x >> 1
            if x % 2 != 0:
                xA ^= A
            MT[i] = MT[(i + M) % N] ^ xA
        index = 0

    # Extract a tempered value based on MT[index]
    var y = MT[index]
    
    y ^= (y >> U) & D
    y ^= (y << S) & B
    y ^= (y << T) & C
    y ^= (y >> L) & D

    index += 1
    
    return y


# Bitmask random_mt to UInt8 
fn random_ui8(inout MT: StaticTuple[624, Int32], inout index: Int) -> UInt8:
    return random_mt(MT, index).value & 0xFF


@value
@register_passable
struct ID(Stringable, CollectionElement):
    var bytes: bytes[16]

    fn __init__(inout self):
        self.bytes = bytes[16]()

    fn __len__(self) -> Int:
        return 16

    fn __setitem__(inout self, index: Int, value: UInt8):
        self.bytes[index] = value

    fn __getitem__(self, index: Int) -> UInt8:
        return self.bytes[index]

    fn __eq__(self, other: Self) -> Bool:
        return self.bytes == other.bytes

    fn __str__(self) -> String:
        return self.formatted_hex()

    fn formatted_hex(self) -> String:
        var result: String = ""
        alias hex_table: String = "0123456789abcdef"
        for i in range(self.__len__()):
            if i==4 or i==6 or i==8 or i==10:
                result += "-"
            result += hex_table[((self.bytes[i] >> 4) & 0xF).to_int()] + hex_table[(self.bytes[i] & 0xF).to_int()]
        return result

@value
struct UUID:
    var MT: StaticTuple[624, Int32]
    var index: Int

    fn __init__(inout self, seed: Int):
        self.MT = mt19937_seed(seed)
        self.index = 624    # Make sure it MT19937 twists on first call

    fn next(inout self) -> ID:
        """
        My version of uuid4.
        """    
        var uuid = ID()
        
        for i in range(16):
            var ui8_value: UInt8 = random_ui8(self.MT, self.index)
            uuid[i] = ui8_value
        
        # Version 4, variant 10xx
        uuid[6] = 0x40 | (0x0F & uuid[6])
        uuid[8] = 0x80 | (0x3F & uuid[8])

        return uuid^