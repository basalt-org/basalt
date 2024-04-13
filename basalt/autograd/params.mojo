from collections.optional import Optional

from basalt import dtype
from basalt import Tensor, TensorShape
from .symbol import Symbol
from .attributes import Attribute


@value
struct Param(CollectionElement, Stringable):
    var data: Optional[List[SIMD[dtype, 1]]]
    var initializer: Optional[Attribute]

    fn __init__(inout self):
        self.data = None
        self.initializer = None

    fn __init__(inout self, data: List[SIMD[dtype, 1]]):
        self.data = data
        self.initializer = None

    fn __init__(inout self, a: SIMD[dtype, 1]):
        var data = List[SIMD[dtype, 1]]()
        data.append(a)
        self.data = data
        self.initializer = None

    fn __init__(inout self, initializer: String, *args: SIMD[dtype, 1]):
        # Supported initializers:
        #   "random_uniform", lower_bound, upper_bound
        #   "random_normal", mean, std
        #   #TODO: "kaiming_uniform", mode, nonlinearity
        #   #TODO: "kaiming_normal", mode, nonlinearity
        self.initializer = Attribute("initializer", initializer)
        var data = List[SIMD[dtype, 1]]()
        for arg in args:
            data.append(arg)
        self.data = data

    fn __getitem__(self, i: Int) -> Optional[SIMD[dtype, 1]]:
        if self.data:
            return self.data.value()[i]
        else:
            return None

    fn __str__(self) -> String:
        var s: String = ""
        if self.data:
            var data = self.data.value()
            s += "["
            for i in range(len(data)):
                s += str(data[i])
                if i < len(data) - 1:
                    s += ", "
            s += "]"
        return s


@value
struct ParamDict(Sized):
    var symbols: List[Symbol]
    var values: List[Param]

    fn __init__(inout self):
        self.symbols = List[Symbol]()
        self.values = List[Param]()

    fn put(inout self, param_id: Symbol, value: Param = Param()):
        self.symbols.append(param_id)
        self.values.append(value)

    fn get_tensor(self, idx: Int) -> Tensor[dtype]:
        # May only be called at runtime
        var num = self.symbols[idx].shape.num_elements()
        var t = DTypePointer[dtype].alloc(num)
        for i in range(num):
            t[i] = self.values[idx][i].value()
        return Tensor[dtype](t, self.symbols[idx].shape)

    fn __len__(self) -> Int:
        return len(self.symbols)
