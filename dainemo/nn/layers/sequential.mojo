from tensor import Tensor


struct Sequential[dtype: DType]:
    '''
    Abstract container of layers
    Sequentially passes the output of one layer the input of the next layer.
    '''

    var layers: VariadicList[AnyType]

    # fn __init__(inout self, *layers: AnyType):
    #     self.layers = layers

    # fn forward(inout self, inputs: Tensor[dtype]):
    #     for i in range(self.layers.__len__()):
    #         outputs = self.layers.__getitem__(i).forward(inputs)
    #         inputs = outputs
    #     return inputs

