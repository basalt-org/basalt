from math import max
from memory.anypointer import AnyPointer

from basalt import Tensor, TensorShape, Symbol


struct Collection(CollectionElement, Sized):
    var size: Int
    var capacity: Int
    var data: AnyPointer[Tensor[dtype]]
    var symbols: DTypePointer[DType.uint32]

    fn __init__(inout self, *, capacity: Int = 0):
        self.size = 0
        self.capacity = capacity
        self.data = AnyPointer[Tensor[dtype]].alloc(capacity)
        self.symbols = DTypePointer[DType.uint32].alloc(capacity)

    fn __moveinit__(inout self, owned existing: Self):
        self.size = existing.size
        self.capacity = existing.capacity
        self.data = existing.data
        self.symbols = existing.symbols

    fn __copyinit__(inout self, existing: Self):
        self = Self(capacity=existing.capacity)
        for i in range(len(existing)):
            self.append((existing.data + i)[], existing.symbols[i])

    @always_inline
    fn __del__(owned self):
        for i in range(self.size):
            _ = (self.data + i).take_value()
        if self.data:
            self.data.free()
        if self.symbols:
            self.symbols.free()

    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn _realloc(inout self, new_capacity: Int):
        var new_data = AnyPointer[Tensor[dtype]].alloc(new_capacity)
        var new_symbols = DTypePointer[DType.uint32].alloc(new_capacity)

        for i in range(self.size):
            (new_data + i).emplace_value((self.data + i).take_value())
            new_symbols[i] = self.symbols[i]

        if self.data:
            self.data.free()
        if self.symbols:
            self.symbols.free()
        
        self.data = new_data
        self.symbols = new_symbols
        self.capacity = new_capacity

    @always_inline
    fn append(inout self, owned value: Tensor[dtype], symbol: Symbol):
        self.append(value^, symbol.name)

    @always_inline
    fn append(inout self, owned value: Tensor[dtype], symbol_name: UInt32):
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        (self.data + self.size).emplace_value(value^)
        self.symbols[self.size] = symbol_name
        self.size += 1

    fn clear(inout self):
        for i in range(self.size):
            _ = (self.data + i).take_value()
            self.symbol[i] = 0
        self.size = 0

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
        self: Reference[Self, mutability, lifetime].mlir_ref_type,
        symbol: Symbol,
    ) -> Reference[Tensor[dtype], mutability, lifetime]:
        var index = Reference(self)[].get_index(symbol.name)

        return Reference(
            __mlir_op.`lit.ref.from_pointer`[
                _type = Reference[Tensor[dtype], mutability, lifetime].mlir_ref_type
            ]((Reference(self)[].data + index).value)
        )

    @always_inline("nodebug")
    fn clear(inout self):
        for i in range(self.size):
            _ = (self.data + i).take_value()
        self.symbols.free()
        self.symbols = DTypePointer[DType.uint32].alloc(self.capacity)
        self.size = 0

    @always_inline("nodebug")
    fn set_zero(self):
        """
        Zeroes out all the tensors in the collection.
        """
        for i in range(self.size):
            self.data[i].zero()