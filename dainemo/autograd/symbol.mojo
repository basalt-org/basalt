from tensor import TensorShape

from dainemo import max_rank
from dainemo.utils.uuid import ID


@value
@register_passable
struct Symbol(CollectionElement, Stringable):
    var name: ID
    var rank: Int
    var dtype: DType
    var static_shape: StaticIntTuple[max_rank]
    var requires_grad: Bool
    var is_constant: Bool

    fn __init__(inout self, name: ID, rank: Int, dtype: DType, static_shape: StaticIntTuple[max_rank], requires_grad: Bool, is_constant: Bool = False):
        self.name = name
        self.rank = rank
        self.dtype = dtype
        self.static_shape = static_shape
        self.requires_grad = requires_grad
        self.is_constant = is_constant
    
    fn __init__(inout self, name: ID, dtype: DType, tensor_shape: TensorShape, requires_grad: Bool, is_constant: Bool = False):
        self.name = name
        self.rank = tensor_shape.rank()
        self.dtype = dtype
        self.static_shape = StaticIntTuple[max_rank]()
        for i in range(tensor_shape.rank()):
            self.static_shape[i] = tensor_shape[i]
        self.requires_grad = requires_grad
        self.is_constant = is_constant

    fn __init__(inout self, name: ID, dtype: DType):
        self.name = name
        self.rank = 1
        self.dtype = dtype
        self.static_shape = StaticIntTuple[max_rank]()
        for i in range(self.rank):
            self.static_shape[i] = 1
        self.requires_grad = False
        self.is_constant = True

    fn __eq__(self, other: Self) -> Bool:
        return self.name == other.name
    
    fn shape(self) -> TensorShape:
        var tmp = DynamicVector[Int]()
        for i in range(self.rank):
            tmp.push_back(self.static_shape[i])
        return TensorShape(tmp)

    fn __str__(self) -> String:
        return self.json()

    fn json(self) -> String:
        return "{\"name\": \"" + str(self.name)[:8] + "\", \"dtype\": \"" + str(self.dtype) + "\", \"shape\": \"" + str(self.shape()) + "\"}"