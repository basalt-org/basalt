from tensor import Tensor
from dainemo.autograd.node import Node
from dainemo.nn.layers import Layer


struct Sequential:
    '''
    Abstract container of layers
    Sequentially passes the output of one layer the input of the next layer.
    '''

    # var layers: VariadicListMem[Layer]
    
    # fn __init__(inout self, *layers: Layer):
    #     self.layers = layers

    # fn __call__(self, inout inputs: Node[dtype]) -> Node[dtype]:
    #     for layer_ptr in self.layers:
    #         inputs = __get_address_as_lvalue(layer_ptr)(inputs)
    #     return inputs

    fn __init__[T: Layer](inout self, layer1: T):
        pass

    fn __init__[T: Layer, T2: Layer](inout self, layer1: T, layer2: T2):
        pass
