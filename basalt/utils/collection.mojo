from memory.anypointer import AnyPointer

from basalt import Tensor, TensorShape, Symbol


struct Collection(Sized):
    var size: Int
    var capacity: Int
    var data: AnyPointer[Tensor[dtype]]
    var symbols: DTypePointer[DType.uint32]

    fn __init__(inout self, *, capacity: Int):
        self.size = 0
        self.capacity = capacity
        self.data = AnyPointer[Tensor[dtype]].alloc(capacity)
        self.symbols = DTypePointer[DType.uint32].alloc(capacity)

    fn __del__(owned self):
        for i in range(self.size):
            _ = (self.data + i).take_value()
        self.data.free()
        self.symbols.free()

    fn __moveinit__(inout self, owned other: Self):
        self.size = other.size
        self.capacity = other.capacity
        self.data = other.data
        self.symbols = other.symbols

    fn append(inout self, owned value: Tensor[dtype], symbol: Symbol):
        # Assumption: Symbol.name contains a unique identifier for the tensor.
        if self.size == self.capacity:
            self.reserve(self.capacity * 2)
        self.data[self.size] = value ^
        self.symbols[self.size] = symbol.name
        self.size += 1

    fn get_index(self, symbol_name: UInt32) -> Int:
        for i in range(self.size):
            if self.symbols[i] == symbol_name:
                return i
        return -1

    # TODO: Check if this can be simplified after #1921 was fixed.
    # Mojo #1921: https://github.com/modularml/mojo/issues/1921#event-12066222345

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

    fn reserve(inout self, new_capacity: Int):
        if new_capacity <= self.capacity:
            return

        var new_data = AnyPointer[Tensor[dtype]].alloc(new_capacity)
        var new_symbols = DTypePointer[DType.uint32].alloc(new_capacity)
        for i in range(self.size):
            (self.data + i).move_into(new_data + i)
            new_symbols[i] = self.symbols[i]

        self.data.free()
        self.symbols.free()
        self.data = new_data
        self.symbols = new_symbols
        self.capacity = new_capacity

    fn clear(inout self):
        for i in range(self.size):
            _ = (self.data + i).take_value()
        self.symbols.free()
        self.symbols = DTypePointer[DType.uint32].alloc(self.capacity)
        self.size = 0

    fn __len__(self) -> Int:
        return self.size

    fn set_zero(self):
        """
        Zeroes out all the tensors in the collection.
        """
        for i in range(self.size):
            self.data[i].zero()
