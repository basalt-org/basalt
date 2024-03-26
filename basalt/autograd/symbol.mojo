from basalt import Tensor, TensorShape


@register_passable("trivial")
struct Symbol(CollectionElement, Stringable):
    var name: UInt32
    var dtype: DType
    var shape: TensorShape
    var trainable: Bool

    fn __init__(
        inout self, name: UInt32, dtype: DType, shape: TensorShape, trainable: Bool
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.trainable = trainable

    fn __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    fn __str__(self) -> String:
        return self.json()

    fn json(self) -> String:
        return (
            '{"name": "'
            + str(self.name)
            + '", "dtype": "'
            + str(self.dtype)
            + '", "shape": "'
            + str(self.shape)
            + '"}'
        )
