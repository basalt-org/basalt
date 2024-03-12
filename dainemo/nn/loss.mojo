from dainemo import Graph, Symbol, OP
from dainemo.nn.activations import LogSoftmax


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


# <------------CROSSENTROPY------------>
fn CrossEntropyLoss(inout g: Graph,
    y_true: Symbol,
    y_pred: Symbol
) -> Symbol:

    # -1/N * sum( targets * log_softmax(outputs) )

    var log_softmax = LogSoftmax(g, y_pred, 1)

    # CrossEntropy (reduction Mean)
    var targets_log_softmax = g.op(OP.MUL, y_true, log_softmax)
    var ret = g.op(OP.SUM, targets_log_softmax)
    var negDivN = g.op(OP.MUL, ret, -1.0 / y_pred.shape()[0])

    return negDivN ^