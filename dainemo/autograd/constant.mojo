from dainemo import max_rank
from dainemo.autograd import Symbol


@value
@register_passable("trivial")
struct Constant[dtype: DType](CollectionElement, Stringable):
    # TODO: To support general constant (instead of only scalars) find a way to store the data for any size!, which is register passable
    # I tried to do it using DTypePointer. This worked and compiled. 
    # But when running it multiple times the compiler cashed and holds only the pointer and the underlying data was overwritten
    # So when trying to initiate the constant param at runtime. It couldn't find the data anymore.
    var data: StaticTuple[1, SIMD[dtype, 1]]
    var rank: Int
    var static_shape: StaticIntTuple[max_rank]

    fn __init__(a: SIMD[dtype, 1]) -> Self:
        var data = StaticTuple[1, SIMD[dtype, 1]](a)
        return Self{data: data, rank: 1, static_shape: StaticIntTuple[max_rank](1)}

    fn __init__(a: Int) -> Self:
        var data = StaticTuple[1, SIMD[dtype, 1]](a)
        return Self{data: data, rank: 1, static_shape: StaticIntTuple[max_rank](1)}

    fn __init__(a: IntLiteral) -> Self:
        var data = StaticTuple[1, SIMD[dtype, 1]](a)
        return Self{data: data, rank: 1, static_shape: StaticIntTuple[max_rank](1)}

    fn __getitem__(self, i: Int) -> SIMD[dtype, 1]:
        return self.data[i]

    fn __setitem__(inout self, i: Int, val: SIMD[dtype, 1]):
        self.data[i] = val

    fn tensor(self) -> Tensor[dtype]:
        # May only be called at runtime
        var tmp = DynamicVector[Int](self.rank)
        for i in range(self.rank):
            tmp.push_back(self.static_shape[i])

        var t = Tensor[dtype](tmp)
        for i in range(t.num_elements()):
            t[i] = self.data[i]
        return t

    fn __str__(self) -> String:
        return str(self.tensor())


@value
struct ConstantDict[dtype: DType](Sized):
    var keys: DynamicVector[Symbol]
    
    var values: DynamicVector[Constant[dtype]]

    fn __init__(inout self):
        self.keys = DynamicVector[Symbol]()
        self.values = DynamicVector[Constant[dtype]]()

    fn put(inout self, key: Symbol, value: Constant[dtype]):
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