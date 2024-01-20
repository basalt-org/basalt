from tensor import Tensor, TensorShape
from math import add, sub, mul, div, log, exp

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import (
    dot,
    tsum,
    tmax,
    elwise_op,
    elwise_pow,
    elwise_transform,
    fill,
    broadcast_elwise_op,
    transpose,
    calculate_strides,
)


"""
Implement forward and backward operations for basic tensor manipulations.
"""


# <------------ADD------------>
struct ADD:
    @staticmethod
    fn forward(n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        """Forward operation of element wise addition."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype]
        if n1.tensor.shape() == n2.tensor.shape():
            res = elwise_op[dtype, nelts, add](n1.tensor, n2.tensor)
        else:
            res = broadcast_elwise_op[dtype, nelts, add](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of element wise addition."""
        # d(x + y) / dx = d(x + y) / dy = 1
        return ug


# <------------SUB------------>
struct SUB:
    @staticmethod
    fn forward(n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        """Forward operation of element wise subtraction."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype]
        if n1.tensor.shape() == n2.tensor.shape():
            res = elwise_op[dtype, nelts, sub](n1.tensor, n2.tensor)
        else:
            res = broadcast_elwise_op[dtype, nelts, sub](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of element wise subtraction."""
        if tensor_id == 0:
            # d(x - y) / dx = 1
            return ug
        else:
            # d(x - y) / dy = -1
            alias nelts = simdwidthof[dtype]()
            let factor: SIMD[dtype, 1] = -1.0
            return elwise_op[dtype, nelts, mul](factor, ug)


# <------------MUL------------>
struct MUL:
    @staticmethod
    fn forward(n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        """Forward operation of element wise multiplication."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype]
        if n1.tensor.shape() == n2.tensor.shape():
            res = elwise_op[dtype, nelts, mul](n1.tensor, n2.tensor)
        else:
            res = broadcast_elwise_op[dtype, nelts, mul](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn forward(n1: Node[dtype], a: SIMD[dtype, 1]) -> Node[dtype]:
        """Forward operation of tensor-scalar multiplication."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, mul](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return GRAPH.create_graph_node[Self.backward](res, n1, Node[dtype](a_tensor))

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of element wise multiplication."""
        alias nelts: Int = simdwidthof[dtype]()
        # d(x*y) / dx = y
        # d(x*y) / dy = x
        let other_id: Int = (tensor_id + 1) % 2
        let other_node = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[other_id])]
        return elwise_op[dtype, nelts, mul](other_node.tensor, ug)


# <------------DIV------------>
struct DIV:
    @staticmethod
    fn forward(n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        """Forward operation of element wise division."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype]
        if n1.tensor.shape() == n2.tensor.shape():
            res = elwise_op[dtype, nelts, div](n1.tensor, n2.tensor)
        else:
            res = broadcast_elwise_op[dtype, nelts, div](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn forward(n1: Node[dtype], a: SIMD[dtype, 1]) -> Node[dtype]:
        """Forward operation of tensor-scalar division."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, div](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return GRAPH.create_graph_node[Self.backward](res, n1, Node[dtype](a_tensor))

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of element wise division."""
        alias nelts: Int = simdwidthof[dtype]()
        # d(x/y) / dx = 1/y
        # d(x/y) / dy = -x/y^2
        if tensor_id == 0:
            let n2 = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])]
            let res = elwise_op[dtype, nelts, div](1.0, n2.tensor)
            return elwise_op[dtype, nelts, mul](res, ug)
        else:
            let n1 = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])]
            let n2 = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])]
            let n2_sq = elwise_pow[dtype, nelts](n2.tensor, 2)
            let div_n1_n2_sq = elwise_op[dtype, nelts, div](n1.tensor, n2_sq)
            let res = elwise_op[dtype, nelts, mul](div_n1_n2_sq, -1.0)
            return elwise_op[dtype, nelts, mul](res, ug)


# <------------DOT------------>
struct DOT:
    @staticmethod
    fn forward(n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        """Forward operation of dot product."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = dot[dtype, nelts](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of dot product."""

        alias nelts: Int = simdwidthof[dtype]()
        if tensor_id == 0:
            let n2 = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])]
            return dot[dtype, nelts](
                ug, transpose[dtype, nelts](n2.tensor)
            )  # dot(ug, n2.T)
        else:
            let n1 = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])]
            return dot[dtype, nelts](
                transpose[dtype, nelts](n1.tensor), ug
            )  # dot(n1.T, ug)


# <------------EXP------------>
struct EXP:
    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        """Forward operation of exp."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_transform[dtype, nelts, exp](n.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of exp."""
        # d(exp(x)) / dx = exp(x)
        alias nelts: Int = simdwidthof[dtype]()
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
        let res = elwise_transform[dtype, nelts, exp](t)
        return elwise_op[dtype, nelts, mul](res, ug)


# <------------LOG------------>
struct LOG:
    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        """Forward operation of log."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_transform[dtype, nelts, log](n.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of log."""
        # d(log(x)) / dx = 1 / x
        alias nelts: Int = simdwidthof[dtype]()
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
        let res = elwise_op[dtype, nelts, div](1.0, t)
        return elwise_op[dtype, nelts, mul](res, ug)


# <------------POW------------>
struct POW:
    @staticmethod
    fn forward(n1: Node[dtype], a: Int) -> Node[dtype]:
        """Forward operation of element wise pow."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_pow[dtype, nelts](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return GRAPH.create_graph_node[Self.backward](res, n1, Node[dtype](a_tensor))

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """Backward operation of element wise pow."""
        # By design: tensor has id = 0 and scalar has id 1
        alias nelts: Int = simdwidthof[dtype]()
        let a: SIMD[dtype, 1] = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])].tensor[0]
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor

        if tensor_id == 0:
            # d(x^y) / dx = y * x^(y-1)
            let res = elwise_op[dtype, nelts, mul](
                a, elwise_pow[dtype, nelts](t, a.to_int() - 1)
            )  # a * t^(a-1)
            return elwise_op[dtype, nelts, mul](res, ug)  # a * t^(a-1) * ug
        else:
            # d(x^y) / dy = x^y * log(x)
            let t_a = elwise_pow[dtype, nelts](t, a.to_int())  # t^a
            let log_t = elwise_transform[dtype, nelts, log](t)  # log(t)
            let res = elwise_op[dtype, nelts, mul](t_a, log_t)  # t^a * log(t)
            return elwise_op[dtype, nelts, mul](res, ug)  # t^a * log(t) * ug


# <------------SUM------------>
struct SUM:
    @staticmethod
    fn forward[axis: Int](n: Node[dtype]) -> Node[dtype]:
        """Forward pass of sum operation: along axis."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = tsum[dtype, nelts](n.tensor, axis=axis)
        return GRAPH.create_graph_node[Self.backward[axis=axis]](res, n)

    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        """Forward pass of sum operation: all elements."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: SIMD[dtype, 1] = tsum[dtype, nelts](n.tensor)
        var res_tensor = Tensor[dtype](1)
        res_tensor[0] = res
        return GRAPH.create_graph_node[Self.backward[axis= -1]](res_tensor, n)

    @staticmethod
    fn backward[
        axis: Int = -1
    ](ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[
        dtype
    ]:
        """Backward pass of sum operation."""
        # TODO: Only 2D input tensors are supported yet !!
        # By design only one node in the collection
        # Output tensor has always the same shape as node input tensor
        alias nelts: Int = simdwidthof[dtype]()
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
        var res = Tensor[dtype](t.shape())
        fill[dtype, nelts](res, 1.0)

        return broadcast_elwise_op[dtype, nelts, mul](res, ug)


# <------------MAX------------>
struct MAX:
    @staticmethod
    fn forward[axis: Int](n: Node[dtype]) -> Node[dtype]:
        """Forward pass of max operation: along axis."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = tmax[dtype, nelts](n.tensor, axis=axis)
        return GRAPH.create_graph_node[Self.backward[axis=axis]](res, n)

    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        """Forward pass of max operation: all elements."""
        alias nelts: Int = simdwidthof[dtype]()
        let res: SIMD[dtype, 1] = tmax[dtype, nelts](n.tensor)
        var res_tensor = Tensor[dtype](1)
        res_tensor[0] = res
        return GRAPH.create_graph_node[Self.backward[axis= -1]](res_tensor, n)

    @staticmethod
    fn backward[
        axis: Int = -1
    ](ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[
        dtype
    ]:
        """Backward pass of max operation."""
        # This could be changed to something like in tinygrad:
        # max_1s = CMPEQ(original_tensor, expanded(max_tensor), axis=axis)
        # sum_max_1s = SUM(max_1s)
        # div_sum_max_1s = DIV(max_1, sum_max_1s)

        # The selected element is 1.0, the others are 0.0. And if there are
        # multiple max values, the gradient is divided by the number of max
        # values (1/n) for each max value.
        alias nelts: Int = simdwidthof[dtype]()
        let t_node = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])]
        let t = t_node.tensor
        let strides = calculate_strides(t.shape())
        var res = Tensor[dtype](t.shape())

        @parameter
        if axis == -1:
            # ug size is 1
            let max_res = tmax[dtype, nelts](t)
            var sum_eq: SIMD[dtype, 1] = 0
            for i in range(t.num_elements()):
                if t[i] == max_res:
                    sum_eq += 1

            let factor = 1 / sum_eq
            for i in range(res.num_elements()):
                if t[i] == max_res:
                    res[i] = factor * ug[0]
        else:
            # max_res.shape == ug.shape
            let max_res = tmax[dtype, nelts](t, axis=axis)

            for i in range(max_res.num_elements()):
                let index_base = (i % strides[axis]) + (i // strides[axis]) * (
                    strides[axis] * t.dim(axis)
                )

                var count_1s: SIMD[dtype, 1] = 0
                # Count the number of values equal to max_res
                for j in range(t.dim(axis)):
                    let index = index_base + j * strides[axis]
                    if t[index] == max_res[i]:
                        count_1s += 1
                # Divide 1.0 by the number of max values (n) and multiply by upper gradient value
                let factor = 1 / count_1s
                for j in range(t.dim(axis)):
                    let index = index_base + j * strides[axis]
                    if t[index] == max_res[i]:
                        res[index] = factor * ug[i]

        return res


# <---------TRANSPOSE--------->
struct TRANSPOSE:
    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        """Forward pass of transpose operation."""
        alias nelts: Int = simdwidthof[dtype]()
        let res = transpose[dtype, nelts](n.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """No local gradient. Transpose is its own inverse."""
        alias nelts: Int = simdwidthof[dtype]()
        return transpose[dtype, nelts](ug)


# <----------FLATTEN---------->
struct FLATTEN:
    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        var res = n.tensor
        try:
            res.ireshape(TensorShape(res.num_elements()))
        except:
            print("[ERROR]: Cannot flatten tensor in forward pass.")

        return GRAPH.create_graph_node[Self.backward](res, n)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """
        Reshape upper gradient to original shape.
        """
        var res = ug
        let shape = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor.shape()

        try:
            res.ireshape(shape)
        except:
            print("[ERROR]: Cannot reshape tensor in flatten backward pass.")

        return res


# <----------RESHAPE---------->
struct RESHAPE:
    @staticmethod
    fn forward(n: Node[dtype], new_shape: TensorShape) -> Node[dtype]:
        var res = n.tensor
        try:
            res.ireshape(new_shape)
        except:
            print("[ERROR]: Cannot reshape tensor in forward pass.")

        return GRAPH.create_graph_node[Self.backward](res, n)

    @staticmethod
    fn backward(
        ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int
    ) -> Tensor[dtype]:
        """
        Reshape upper gradient to original shape.
        """
        var res = ug
        let shape = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor.shape()

        try:
            res.ireshape(shape)
        except:
            print("[ERROR]: Cannot reshape tensor in reshape backward pass.")

        return res
