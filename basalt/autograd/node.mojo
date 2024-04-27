from collections.optional import Optional
from utils.variant import Variant

from basalt.autograd import Symbol
from basalt.autograd.ops import OP

from .attributes import AttributeVector


@value
struct Node(CollectionElement, Stringable):
    var operator: OP
    var inputs: List[Symbol]
    var outputs: List[Symbol]
    var attributes: AttributeVector

    fn __init__(
        inout self,
        operator: OP,
        inputs: List[Symbol],
        outputs: List[Symbol],
        attributes: AttributeVector = AttributeVector(),
    ):
        self.operator = operator
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes

    fn __str__(self) -> String:
        return self.json()

    fn json(self) -> String:
        var s: String = '{"operator": "' + str(self.operator.name) + '", "inputs": ['
        for i in range(len(self.inputs)):
            s += self.inputs[i].json()
            if i < len(self.inputs) - 1:
                s += ", "
        s += '], "outputs": ['
        for i in range(len(self.outputs)):
            s += self.outputs[i].json()
            if i < len(self.outputs) - 1:
                s += ", "
        s += "]}"
        return s
