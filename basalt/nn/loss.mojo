import basalt.nn as nn
from basalt import Tensor, TensorShape
from basalt import Graph, Symbol, OP



fn MSELoss(inout g: Graph,
    y_pred: Symbol,
    y_true: Symbol,
) -> Symbol:

    # 1/N * sum( (outputs - targets)^2 )

    var diff = g.op(OP.SUB, y_true, y_pred)
    var loss = g.op(OP.POW, diff, 2)
    var mean_loss = g.op(OP.MEAN, loss)

    return mean_loss


fn CrossEntropyLoss(inout g: Graph,
    y_pred: Symbol,
    y_true: Symbol,
) -> Symbol:

    # -1/N * sum( targets * log_softmax(outputs) )

    var log_softmax = nn.LogSoftmax(g, y_pred, axis=1)

    # CrossEntropy (reduction Mean)
    var targets_log_softmax = g.op(OP.MUL, y_true, log_softmax)
    var ret = g.op(OP.SUM, targets_log_softmax)
    var negDivN = g.op(OP.MUL, ret, -1.0 / y_pred.shape[0])

    return negDivN