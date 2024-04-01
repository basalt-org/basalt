from basalt import Tensor, TensorShape


@register_passable("trivial")
struct Symbol(CollectionElement, Stringable):
    var name: UInt32
    var dtype: DType
    var shape: TensorShape
    var trainable: Bool

    @always_inline("nodebug")
    fn __init__(
        inout self, name: UInt32, dtype: DType, shape: TensorShape, trainable: Bool
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.trainable = trainable

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return self.json()

    @always_inline("nodebug")
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
