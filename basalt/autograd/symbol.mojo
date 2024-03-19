from tensor import TensorShape

from basalt import max_rank
from basalt.utils.uuid import UUID


@value
@register_passable
struct Symbol(CollectionElement, Stringable):
    var name: UUID
    var rank: Int
    var dtype: DType
    var static_shape: StaticIntTuple[max_rank]
    var trainable: Bool
    
    fn __init__(inout self, name: UUID, dtype: DType, tensor_shape: TensorShape, trainable: Bool):
        self.name = name
        self.rank = tensor_shape.rank()
        self.dtype = dtype
        self.static_shape = StaticIntTuple[max_rank]()
        for i in range(tensor_shape.rank()):
            self.static_shape[i] = tensor_shape[i]
        self.trainable = trainable

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