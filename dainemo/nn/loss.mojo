from tensor import Tensor


# <------------MSE------------>
struct MSE[dtype: DType]:
    def __init__(inout self):
        pass

    def forward(self, outputs: Tensor[dtype], targets: Tensor[dtype]):
        '''Forward pass of MSE.'''
        pass

        # num_examples = outputs.dim(0)
        # cost = (1/(2*num_examples))*_sum((outputs-targets)**2)
        # return cost

    # def __call__(self, outputs: Tensor[dtype], targets: Tensor[dtype]):
    #     return self.forward(outputs, targets)


# <------------BINARYCROSSENTROPY------------>
# TODO

# <------------CROSSENTROPY------------>
# TODO

# <------------SOFTMAXCROSSENTROPY------------>
# TODO
