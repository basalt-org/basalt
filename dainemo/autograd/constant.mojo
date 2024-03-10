from dainemo import max_rank, dtype
from dainemo.autograd import Symbol


@value
struct Constant(CollectionElement, Stringable):
    var rank: Int
    var static_shape: StaticIntTuple[max_rank]
    var data: DynamicVector[SIMD[dtype, 1]]

    fn __init__(inout self, a: SIMD[dtype, 1]):
        self.rank = 1
        self.static_shape = StaticIntTuple[max_rank](1)
        self.data = DynamicVector[SIMD[dtype, 1]]()
        self.data.push_back(a)

    fn __init__(inout self, a: Int):
        self.rank = 1
        self.static_shape = StaticIntTuple[max_rank](1)
        self.data = DynamicVector[SIMD[dtype, 1]]()
        self.data.push_back(a)

    fn __init__(inout self, a: IntLiteral):
        self.rank = 1
        self.static_shape = StaticIntTuple[max_rank](1)
        self.data = DynamicVector[SIMD[dtype, 1]]()
        self.data.push_back(a)

    fn __getitem__(self, i: Int) -> SIMD[dtype, 1]:
        return self.data[i]

    fn __setitem__(inout self, i: Int, val: SIMD[dtype, 1]):
        self.data[i] = val

    fn tensor(self) -> Tensor[dtype]:
        # May only be called at runtime
        var tmp = DynamicVector[Int]()
        for i in range(self.rank):
            tmp.push_back(self.static_shape[i])

        var t = Tensor[dtype](tmp)
        for i in range(t.num_elements()):
            t[i] = self.data[i]
        return t

    fn __str__(self) -> String:
        return str(self.tensor())


@value
struct ConstantDict(Sized):
    var keys: DynamicVector[Symbol]
    
    var values: DynamicVector[Constant]

    fn __init__(inout self):
        self.keys = DynamicVector[Symbol]()
        self.values = DynamicVector[Constant]()

    fn put(inout self, key: Symbol, value: Constant):
        self.keys.push_back(key)
        self.values.push_back(value)

    fn get(self, key: Symbol) -> Tensor[dtype]:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.values[i].tensor()
        print("[ERROR] Key not found in ConstantDict")
        return Tensor[dtype]()

    fn __len__(self) -> Int:
        return len(self.keys)