from tensor import TensorShape
from collections.optional import Optional

from dainemo import max_rank, dtype
from dainemo.autograd import Symbol


@value
struct Param(CollectionElement, Stringable):
    var data: DynamicVector[SIMD[dtype, 1]]

    fn __init__(inout self):
        self.data = DynamicVector[SIMD[dtype, 1]]()
    
    fn __init__(inout self, data: DynamicVector[SIMD[dtype, 1]]):
        self.data = data

    fn __init__(inout self, a: SIMD[dtype, 1]):
        self.data = DynamicVector[SIMD[dtype, 1]]()
        self.data.push_back(a)

    fn __init__(inout self, a: Int):
        self.data = DynamicVector[SIMD[dtype, 1]]()
        self.data.push_back(a)

    fn __init__(inout self, a: IntLiteral):
        self.data = DynamicVector[SIMD[dtype, 1]]()
        self.data.push_back(a)

    fn __getitem__(self, i: Int) -> SIMD[dtype, 1]:
        return self.data[i]

    fn __setitem__(inout self, i: Int, val: SIMD[dtype, 1]):
        self.data[i] = val

    fn __str__(self) -> String:
        var s: String = "["
        for i in range(len(self.data)):
            s += str(self.data[i])
            if i < len(self.data) - 1:
                s += ", "
        return s + "]"


@value
struct ParamDict(Sized):
    var symbols: DynamicVector[Symbol]
    var initialized: DynamicVector[Bool]
    var values: DynamicVector[Param]

    fn __init__(inout self):
        self.symbols = DynamicVector[Symbol]()
        self.initialized = DynamicVector[Bool]()
        self.values = DynamicVector[Param]()

    fn put(inout self, param_id: Symbol, value: Optional[Param] = None):
        self.symbols.push_back(param_id)
        if value:
            # Initialized parameter
            self.initialized.push_back(True)
            self.values.push_back(value.value())
        else:
            # Uninitialized parameter
            self.initialized.push_back(False)
            self.values.push_back(Param())

    fn get_tensor(self, idx: Int) -> Tensor[dtype]:
        # May only be called at runtime
        var t = Tensor[dtype](self.symbols[idx].shape())
        for i in range(t.num_elements()):
            t[i] = self.values[idx][i]
        return t ^

    fn __len__(self) -> Int:
        return len(self.symbols)