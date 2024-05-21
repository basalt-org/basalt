from math import max, divmod
from memory.unsafe_pointer import UnsafePointer, initialize_pointee_move, destroy_pointee

from basalt import Tensor, Symbol


struct Collection(CollectionElement, Sized):
    """
    A collection of tensors with associated symbols.
    """

    var size: Int
    var capacity: Int
    var data: UnsafePointer[Tensor[dtype]]
    var symbols: DTypePointer[DType.uint32]

    @always_inline("nodebug")
    fn __init__(inout self, *, capacity: Int = 0):
        """
        Initializes a new Collection with the given capacity.
        """
        self.size = 0
        self.capacity = capacity
        self.data = UnsafePointer[Tensor[dtype]].alloc(capacity)
        self.symbols = DTypePointer[DType.uint32].alloc(capacity)

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned existing: Self):
        """
        Move initializes a Collection from an existing one.
        """
        self.size = existing.size
        self.capacity = existing.capacity
        self.data = existing.data
        self.symbols = existing.symbols

    @always_inline("nodebug")
    fn __copyinit__(inout self, existing: Self):
        """
        Copy initializes a Collection from an existing one.
        """
        self.capacity = existing.capacity
        self.size = existing.size
        self.data = UnsafePointer[Tensor[dtype]].alloc(existing.capacity)
        self.symbols = DTypePointer[DType.uint32].alloc(existing.capacity)
        memcpy(self.symbols, existing.symbols, existing.capacity)

        for i in range(existing.size):
            initialize_pointee_move((self.data + i), (existing.data + i)[])

    @always_inline("nodebug")
    fn __del__(owned self):
        """
        Destructor for the Collection.
        """
        for i in range(self.size):
            destroy_pointee((self.data + i))
        if self.data:
            self.data.free()
        if self.symbols:
            self.symbols.free()

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """
        Returns the number of elements in the Collection.
        """
        return self.size

    @always_inline("nodebug")
    fn _realloc(inout self, new_capacity: Int):
        """
        Reallocates the Collection to the new capacity.
        """
        var new_data = UnsafePointer[Tensor[dtype]].alloc(new_capacity)
        var new_symbols = DTypePointer[DType.uint32].alloc(new_capacity)

        for i in range(self.size):
            initialize_pointee_move((new_data + i), (self.data + i)[])
            new_symbols[i] = self.symbols[i]

        self.data.free()
        self.symbols.free()

        self.data = new_data
        self.symbols = new_symbols
        self.capacity = new_capacity

    @always_inline("nodebug")
    fn append(inout self, owned value: Tensor[dtype], symbol: Symbol):
        """
        Appends a tensor and its associated symbol to the Collection.
        """
        self.append(value ^, symbol.name)

    @always_inline("nodebug")
    fn append(inout self, owned value: Tensor[dtype], symbol_name: UInt32):
        """
        Appends a tensor and its associated symbol name to the Collection.
        """
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        initialize_pointee_move((self.data + self.size), value ^)
        self.symbols[self.size] = symbol_name
        self.size += 1

    @always_inline("nodebug")
    fn get_index(self, symbol_name: UInt32) -> Int:
        """
        Returns the index of the tensor with the given symbol name.
        """        
        alias factor = 8
        # 2 -> 5.32s MNIST
        # 4 -> 4.95s MNIST
        # 8 -> 4.85s MNIST
        # 16 -> 5.19s MNIST
        # NOTE: This ideally should just be a hashmap

        for i in range(0, self.size, factor):
            var elems = self.symbols.load[width=factor](i) == symbol_name

            for j in range(factor):
                if elems[j]: 
                    return i + j

        var split = divmod(self.size, factor)

        for i in range(split[1]):
            var index = split[0] + i
            
            if self.symbols[index] == symbol_name:
                return index

        return -1

    @always_inline("nodebug")
    fn __refitem__[
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](
        self: Reference[Self, mutability, lifetime]._mlir_type,
        symbol: Symbol,
    ) -> Reference[Tensor[dtype], mutability, lifetime]:
        """
        Returns a reference to the tensor with the given symbol.
        """
        var index = Reference(self)[].get_index(symbol.name)

        return (Reference(self)[].data + index)[]

    @always_inline("nodebug")
    fn clear(inout self):
        """
        Clears the Collection, removing all tensors and symbols.
        """
        for i in range(self.size):
            destroy_pointee((self.data + i))
        memset_zero(self.symbols, self.capacity)
        self.size = 0

    @always_inline("nodebug")
    fn set_zero(self):
        """
        Zeroes out all the tensors in the collection.
        """
        for i in range(self.size):
            self.data[i].zero()
