from dainemo import Graph, Symbol, OP


# <------------MSE------------>
fn MSELoss(inout g: Graph,
    y_true: Symbol,
    y_pred: Symbol
) -> Symbol:

    # 1/N * sum( (outputs - targets)^2 )

    var diff = g.op(OP.SUB, y_true, y_pred)
    var loss = g.op(OP.POW, diff, 2)
    var mean_loss = g.op(OP.MEAN, loss)

    return mean_loss ^


# # <------------CROSSENTROPY------------>
# struct CrossEntropyLoss:
#     fn __init__(inout self):
#         pass

#     fn forward(inout self, inout outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
#         """
#         Forward pass of CrossEntropy.
#         Epsilons is a small value for numerical stability to prevent log(0).
#         """
#         # -1/N * sum( targets * log_softmax(outputs) )
        
#         # LogSoftmax
#         var act = nn.activations.LogSoftmax[axis=1]()
#         var log_softmax = act(outputs)
        
#         # CrossEntropy (reduction Mean)
#         var targets_log_softmax = MUL.forward(Node[dtype](targets), log_softmax)
#         var ret = SUM.forward(targets_log_softmax)
#         var negDivN: SIMD[dtype, 1] = (-1/outputs.tensor.dim(0)).cast[dtype]()
#         return MUL.forward(ret, negDivN)

#     fn __call__(inout self, inout outputs: Node[dtype], targets: Tensor[dtype]) -> Node[dtype]:
#         return self.forward(outputs, targets)
