from memory.anypointer import AnyPointer

from basalt import Tensor, TensorShape, Symbol
from .string_dict import StringDict


struct Collection(Sized):
    var data: AnyPointer[Tensor[dtype]]
    var size: Int
    var capacity: Int
    var symbol_map: StringDict[Int]

    @always_inline
    fn __init__(inout self, *, capacity: Int):
        self.capacity = capacity
        self.data = AnyPointer[Tensor[dtype]].alloc(capacity)
        self.size = 0
        self.symbol_map = StringDict[Int]()

    @always_inline
    fn __del__(owned self):
        for i in range(self.size):
            _ = (self.data + i).take_value()
        self.data.free()

    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        self.capacity = other.capacity
        self.size = other.size
        self.data = other.data
        self.symbol_map = other.symbol_map^

    fn append(inout self, owned value: Tensor[dtype], symbol: Symbol):
        if self.size == self.capacity:
            self.reserve(self.capacity * 2)
        self.data[self.size] = value ^
        self.symbol_map.put(str(symbol.name), self.size)
        self.size += 1

    # TODO: Check if this can be simplified after #1921 was fixed.
    # Mojo #1921: https://github.com/modularml/mojo/issues/1921#event-12066222345
    fn __refitem__[
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](
        self: Reference[Self, mutability, lifetime].mlir_ref_type,
        symbol: Symbol,
    ) -> Reference[Tensor[dtype], mutability, lifetime]:
        
        var index = Reference(self)[].symbol_map.get(str(symbol.name), -1)
        
        return Reference(
            __mlir_op.`lit.ref.from_pointer`[
                _type = Reference[Tensor[dtype], mutability, lifetime].mlir_ref_type
            ]((Reference(self)[].data + index).value)
        )

    @always_inline
    fn reserve(inout self, new_capacity: Int):
        if new_capacity <= self.capacity:
            return
        
        var new_data = AnyPointer[Tensor[dtype]].alloc(new_capacity)
        for i in range(self.size):
            (self.data + i).move_into(new_data + i)
        
        self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    @always_inline
    fn clear(inout self):
        for i in range(self.size):
            _ = (self.data + i).take_value()
        self.size = 0

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn set_zero(self):
        """
        Zeroes out all the tensors in the collection.
        """
        for i in range(self.size):
            self.data[i].zero()