from tensor import Tensor
from dainemo.autograd.graph import Graph
from dainemo.utils.tensorutils import dot, tsum

'''
Implement forward and backward operations for basic tensor manipulations.
'''

# <------------ADD------------>
struct Add:
    pass


# <------------SUB------------>
# TODO


# <------------MUL------------>
# TODO


# <------------DIV------------>
# TODO


# <------------DOT------------>
struct DOT[dtype: DType]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
        '''Forward operation of dot product.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res = dot[dtype, nelts](t1, t2)
        graph.set_forward_op(res, t1, t2)
        return res

    @staticmethod
    fn backward(inout graph: Graph[dtype], t1: Tensor[dtype], t2: Tensor[dtype]):
        '''Backward operation of dot product.'''
        # TODO
        pass



# <------------EXP------------>
# TODO


# <------------LOG------------>
# TODO


# <------------POW------------>
# TODO


# <------------SUM------------>
struct SUM[dtype: DType, axis: Int = 0]:
    @staticmethod
    fn forward(inout graph: Graph[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        '''Forward pass of sum operation.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = tsum[dtype, nelts](t, axis=axis)
        graph.set_forward_op(res, t)
        return res

    @staticmethod
    fn backward(inout graph: Graph[dtype], t: Tensor[dtype]):
        '''Backward pass of sum operation.'''
        # TODO
        pass

# <---------TRANSPOSE--------->
# TODO


# <----------FLATTEN---------->
# TODO


# <----------RESHAPE---------->
# TODO

