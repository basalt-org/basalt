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

    fn __init__(name: ID, rank: Int, dtype: DType, static_shape: StaticIntTuple[max_rank], requires_grad: Bool, is_constant: Bool = False) -> Self:
        return Self{name: name, rank: rank, dtype: dtype, static_shape: static_shape, requires_grad: requires_grad, is_constant: is_constant}
    
    fn __init__(name: ID, dtype: DType, tensor_shape: TensorShape, requires_grad: Bool, is_constant: Bool = False) -> Self:
        var static_shape = StaticIntTuple[max_rank]()
        for i in range(tensor_shape.rank()):
            static_shape[i] = tensor_shape[i]

        return Self{name: name, rank: tensor_shape.rank(), dtype: dtype, static_shape: static_shape, requires_grad: requires_grad, is_constant: is_constant}

    fn __init__(name: ID, dtype: DType) -> Self:
        var rank = 1
        var static_shape = StaticIntTuple[max_rank]()
        for i in range(rank):
            static_shape[i] = 1

        return Self{name: name, rank: rank, dtype: dtype, static_shape: static_shape, requires_grad: False, is_constant: True}

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