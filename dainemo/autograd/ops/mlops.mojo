from tensor import Tensor, TensorShape
from math import sub, mul, exp, max

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import (
    elwise_op,
    elwise_transform,
    fill,
    batch_tensor_elwise_op,
)


# --------------UNARY OPERATORS----------------


# <------------SIGMOID------------>
struct SIGMOID:
    @staticmethod
    fn sigmoid[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return 1 / (1 + exp(-x))

    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        """Forward operation of sigmoid."""
        alias nelts = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_transform[dtype, nelts, SIGMOID.sigmoid](
            n.tensor
        )
        return GRAPH.create_graph_node[Self.backward](res, n)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of sigmoid."""
        # d(sigmod(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        alias nelts = simdwidthof[dtype]()
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
        # sigmoid(x)
        let sigmoid_res = elwise_transform[dtype, nelts, SIGMOID.sigmoid](t)
        # 1 - sigmoid(x)
        let sub_res = elwise_op[dtype, nelts, sub](SIMD[dtype, 1](1), sigmoid_res)
        # sigmoid(x) * (1 - sigmoid(x))
        let res = elwise_op[dtype, nelts, mul](sigmoid_res, sub_res)

        return elwise_op[dtype, nelts, mul](ug, res)


# <------------RELU------------>
struct RELU:
    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        """Forward operation of relu."""
        alias nelts = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, max](
            n.tensor, SIMD[dtype, 1](0)
        )
        return GRAPH.create_graph_node[Self.backward](res, n)

    @staticmethod
    fn relu_derivative[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return 1 if x > 0 else 0

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        pass
        # d(relu(x))/dx = 1 if x > 0 else 0. We also give 0 to x = 0 instead of undefined.
        alias nelts = simdwidthof[dtype]()
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
        let res = elwise_transform[dtype, nelts, RELU.relu_derivative](t)

        return elwise_op[dtype, nelts, mul](ug, res)
