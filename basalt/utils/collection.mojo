from math import max

from basalt.nn import Tensor, TensorShape
from basalt.autograd import Symbol


struct Collection:
    var size: Int
    var capacity: Int
    var data: UnsafePointer[Tensor[dtype]]
    var symbols: DTypePointer[DType.uint32]

    @always_inline("nodebug")
    fn __init__(inout self, *, capacity: Int = 0):
        self.size = 0
        self.capacity = capacity
        self.data = UnsafePointer[Tensor[dtype]].alloc(capacity)
        self.symbols = DTypePointer[DType.uint32].alloc(capacity)

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        self.data = existing.data
        self.symbols = existing.symbols

    @always_inline("nodebug")
    fn __copyinit__(inout self, existing: Self):
        self.capacity = existing.capacity
        self.size = existing.size
        self.data = UnsafePointer[Tensor[dtype]].alloc(existing.capacity)
        self.symbols = DTypePointer[DType.uint32].alloc(existing.capacity)

        for i in range(existing.size):
            initialize_pointee_move((self.data + i), (existing.data + i)[])

        memcpy(self.symbols, existing.symbols, existing.capacity)

    @always_inline("nodebug")
    fn __del__(owned self):
        if self.data:
            for i in range(self.size):
                destroy_pointee((self.data + i))
            self.data.free()
        if self.symbols:
            self.symbols.free()

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.size

    @always_inline("nodebug")
    fn reserve(inout self, capacity: Int):
        self.data = UnsafePointer[Tensor[dtype]].alloc(capacity)
        self.symbols = DTypePointer[DType.uint32].alloc(capacity)
        self.capacity = capacity

    @always_inline("nodebug")
    fn _realloc(inout self, new_capacity: Int):
        var new_data = UnsafePointer[Tensor[dtype]].alloc(new_capacity)
        var new_symbols = DTypePointer[DType.uint32].alloc(new_capacity)

        for i in range(self.size):
            initialize_pointee_move((new_data + i), (self.data + i)[])

        memcpy(new_symbols, self.symbols, self.capacity)

        self.data.free()
        self.symbols.free()

        self.data = new_data
        self.symbols = new_symbols
        self.capacity = new_capacity

    @always_inline("nodebug")
    fn append(inout self, owned value: Tensor[dtype], symbol: Symbol):
        self.append(value ^, symbol.name)

    @always_inline("nodebug")
    fn append(inout self, owned value: Tensor[dtype], symbol_name: UInt32):
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        initialize_pointee_move((self.data + self.size), value ^)
        self.symbols[self.size] = symbol_name
        self.size += 1

    @always_inline("nodebug")
    fn get_index(self, symbol_name: UInt32) -> Int:
        for i in range(self.size):
            if self.symbols[i] == symbol_name:
                return i
        return -1

    @always_inline("nodebug")
    fn __refitem__[
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](
        self: Reference[Self, mutability, lifetime]._mlir_type,
        symbol: Symbol,
    ) -> Reference[Tensor[dtype], mutability, lifetime]:
        var index = Reference(self)[].get_index(symbol.name)

        return (Reference(self)[].data + index)[]

    @always_inline("nodebug")
    fn clear(inout self):
        for i in range(self.size):
            destroy_pointee((self.data + i))
        memset_zero(self.symbols, self.capacity)
        self.size = 0

    @always_inline("nodebug")
    fn set_zero(self):
        for i in range(self.size):
            self.data[i].zero()
