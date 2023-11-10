from tensor import Tensor
from dainemo.autograd.node import Node
from dainemo.autograd.graph import Graph
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
        return graph.create_graph_node["bw_ADD"](res, n1, n2)           # "bw_ADD"  --> ADD[dtype].backward

    @staticmethod
    fn backward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]):
        '''Backward operation of element wise addition.'''
        pass


# <------------SUB------------>
struct SUB[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of element wise subtraction.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, sub](n1.tensor, n2.tensor)
        return graph.create_graph_node["bw_SUB"](res, n1, n2)           # "bw_SUB"  --> SUB[dtype].backward

    @staticmethod
    fn backward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]):
        '''Backward operation of element wise subtraction.'''
        pass


# <------------MUL------------>
struct MUL[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of element wise multiplication.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, mul](n1.tensor, n2.tensor)
        return graph.create_graph_node["bw_MUL"](res, n1, n2)           # "bw_MUL"  --> SUB[dtype].backward

    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], a: SIMD[dtype, 1]) -> Node[dtype]:
        '''Forward operation of tensor-scalar multiplication.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, mul](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return graph.create_graph_node["bw_MUL_scalar"](res, n1, Node[dtype](a_tensor))    # "bw_MUL_scalar"  --> MUL[dtype].backward

    @staticmethod
    fn backward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]):
        '''Backward operation of element wise multiplication.'''
        pass

# <------------DIV------------>
# TODO


# <------------DOT------------>
struct DOT[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of dot product.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = dot[dtype, nelts](n1.tensor, n2.tensor)
        return graph.create_graph_node["bw_DOT"](res, n1, n2)           # "bw_DOT"  --> DOT[dtype].backward

    @staticmethod
    fn backward(inout graph: Graph[dtype], n1: Node[dtype], n2: Node[dtype]):
        '''Backward operation of dot product.'''
        # TODO: sets the grad_fn of the input tensors
        pass



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
        return graph.create_graph_node["bw_POW"](res, n1, Node[dtype](a_tensor))           # "bw_POW"  --> POW[dtype].backward


# <------------SUM------------>
struct SUM[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], n: Node[dtype], axis: Int) -> Node[dtype]:
        '''Forward pass of sum operation: along axis.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = tsum[dtype, nelts](n.tensor, axis=axis)
        return graph.create_graph_node["bw_SUM_axis"](res, n)   # "bw_SUM_axis"  --> SUM[dtype].backward

    @staticmethod
    fn forward(inout graph: Graph[dtype], n: Node[dtype]) -> Node[dtype]:
        '''Forward pass of sum operation: all elements.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: SIMD[dtype, 1] = tsum[dtype, nelts](n.tensor)
        var res_tensor = Tensor[dtype](1)
        res_tensor[0] = res
        return graph.create_graph_node["bw_SUM_all"](res_tensor, n)    # "bw_SUM_all"  --> SUM[dtype].backward

    @staticmethod
    fn backward(inout graph: Graph[dtype], n: Node[dtype]):
        '''Backward pass of sum operation.'''
        # TODO: sets the grad_fn of the input tensors
        pass

# <---------TRANSPOSE--------->
# TODO


# <----------FLATTEN---------->
# TODO


# <----------RESHAPE---------->
# TODO

