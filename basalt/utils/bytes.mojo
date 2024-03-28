from math import min


@value
@register_passable("trivial")
struct bytes[capacity: Int](Stringable, CollectionElement):
    """
    Static sequence of bytes.
    """

    var _vector: StaticTuple[UInt8, capacity]

    fn __init__(inout self):
        var _vector = StaticTuple[UInt8, capacity]()
        for i in range(capacity):
            _vector[i] = 0
        self._vector = _vector

    fn __init__(inout self, s: String):
        var _vector = StaticTuple[UInt8, capacity]()
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
            result += (
                hex_table[((self[i] >> 4) & 0xF).to_int()]
                + hex_table[(self[i] & 0xF).to_int()]
            )
        return result
