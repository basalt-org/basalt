import random


struct bytes:
    '''
    Dynamic sequence of bytes
    '''
    var _vector: DynamicVector[UInt8]

    fn __init__(inout self, capacity: Int):
        self._vector = DynamicVector[UInt8](capacity=capacity)
        for i in range(capacity):
            self._vector.push_back(0)

    fn __len__(self) -> Int:
        return len(self._vector)

    fn __setitem__(inout self, index: Int, value: UInt8):
        self._vector[index] = value

    fn __getitem__(inout self, index: Int) -> UInt8:
        return self._vector[index]

    fn hex(inout self) -> String:
        var result: String = ""
        alias hex_table: String = "0123456789abcdef"
        for i in range(self.__len__()):
            if i==4 or i==6 or i==8 or i==10:
                result += "-"
            result += hex_table[(self[i] >> 4).to_int()] + hex_table[(self[i] & 0xF).to_int()]
        return result


fn uuid() -> String:
    '''
    My version of uuid4
    '''
    var uuid: bytes = bytes(capacity=16)
    
    # (Mojo v0.5.0) Can only find the ability of generating random_ui64
    # Adapt the range to 0..2^8-1 to fit the type of UInt8
    for i in range(16):
        let ui64_value: UInt64 = random.random_ui64(0, 255)
        let ui8_value: UInt8 = ui64_value.cast[DType.uint8]()

        uuid[i] = ui8_value
    
    # Version 4, variant 10xx
    uuid[6] = 0x40 | (0x0F & uuid[6])
    uuid[8] = 0x80 | (0x3F & uuid[8])

    return uuid.hex()