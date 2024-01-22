from dainemo.autograd.node import Node

trait Layer:
    fn forward(self, inputs: Node[dtype]) -> Node[dtype]:
        ...

    fn __call__(self, inputs: Node[dtype]) -> Node[dtype]:
        ...