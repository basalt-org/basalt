from tensor import Tensor
from dainemo.autograd.node import Node
from dainemo.autograd.graph import Graph
from dainemo.utils.collection import NodeCollection
from dainemo.utils.tensorutils import dot, tsum, elwise_op, elwise_pow

from math import add, sub, mul, div

'''
Implement forward and backward operations for basic tensor manipulations.
'''

# <------------ADD------------>
struct ADD[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of element wise addition.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, add](n1.tensor, n2.tensor)
        return graph.create_graph_node[Self.backward[dtype]](res, n1, n2)

    @staticmethod
    fn backward[dtype: DType](ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise addition.'''
        return ug


# <------------SUB------------>
struct SUB[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of element wise subtraction.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, sub](n1.tensor, n2.tensor)
        return graph.create_graph_node[Self.backward[dtype]](res, n1, n2)

    @staticmethod
    fn backward[dtype: DType](ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise subtraction.'''
        if node_id == 0:
            return ug
        else:
            alias nelts = simdwidthof[dtype]()
            let factor: SIMD[dtype, 1] = -1.0
            return elwise_op[dtype, nelts, mul](factor, ug)


# <------------MUL------------>
struct MUL[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of element wise multiplication.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, mul](n1.tensor, n2.tensor)
        return graph.create_graph_node[Self.backward[dtype]](res, n1, n2)

    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], a: SIMD[dtype, 1]) -> Node[dtype]:
        '''Forward operation of tensor-scalar multiplication.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, mul](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return graph.create_graph_node[Self.backward[dtype]](res, n1, Node[dtype](a_tensor))

    @staticmethod
    fn backward[dtype: DType](ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise multiplication.'''
        alias nelts: Int = simdwidthof[dtype]()
        let other_id: Int = (node_id + 1) % 2
        return elwise_op[dtype, nelts, mul](nodes.get(other_id).tensor, ug)


# <------------DIV------------>
# TODO


# <------------DOT------------>
struct DOT[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of dot product.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = dot[dtype, nelts](n1.tensor, n2.tensor)
        return graph.create_graph_node[Self.backward[dtype]](res, n1, n2)

    @staticmethod
    fn backward[dtype: DType](ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]:
        '''Backward operation of dot product.'''
        # TODO: sets the grad_fn of the input tensors
        print("DOT backward")
        return Tensor[dtype](ug.shape())



# <------------EXP------------>
# TODO


# <------------LOG------------>
# TODO


# <------------POW------------>
struct POW[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], a: Int) -> Node[dtype]:
        '''Forward operation of element wise pow.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_pow[dtype, nelts](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return graph.create_graph_node[Self.backward[dtype]](res, n1, Node[dtype](a_tensor))

    @staticmethod
    fn backward[dtype: DType](ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise pow.'''
        print("POW backward")
        return Tensor[dtype](ug.shape())


# <------------SUM------------>
struct SUM[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n: Node[dtype], axis: Int) -> Node[dtype]:
        '''Forward pass of sum operation: along axis.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = tsum[dtype, nelts](n.tensor, axis=axis)
        return graph.create_graph_node[Self.backward[dtype]](res, n)

    @staticmethod
    fn forward(inout graph: Graph[dtype], n: Node[dtype]) -> Node[dtype]:
        '''Forward pass of sum operation: all elements.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: SIMD[dtype, 1] = tsum[dtype, nelts](n.tensor)
        var res_tensor = Tensor[dtype](1)
        res_tensor[0] = res
        return graph.create_graph_node[Self.backward[dtype]](res_tensor, n)

    @staticmethod
    fn backward[dtype: DType](ug: Tensor[dtype], nodes: NodeCollection[dtype], node_id: Int) -> Tensor[dtype]:
        '''Backward pass of sum operation.'''
        # TODO: sets the grad_fn of the input tensors
        print("SUM backward")
        return Tensor[dtype](ug.shape())

# <---------TRANSPOSE--------->
# TODO


# <----------FLATTEN---------->
# TODO


# <----------RESHAPE---------->
# TODO

