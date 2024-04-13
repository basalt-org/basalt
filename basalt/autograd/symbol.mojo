from basalt import Tensor, TensorShape


@value
@register_passable("trivial")
struct Symbol(CollectionElement, Stringable, EqualityComparable):
    var name: UInt32
    var dtype: DType
    var shape: TensorShape
    var trainable: Bool

    fn __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    fn __ne__(self, other: Self) -> Bool:
        return self.name != other.name

    fn __str__(self) -> String:
        return (
            '{"name": "'
            + str(self.name)
            + '", "dtype": "'
            + str(self.dtype)
            + '", "shape": "'
            + str(self.shape)
            + '"}'
        )

    fn json(self) -> String:
        return str(self)
