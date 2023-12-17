from tensor import Tensor, TensorShape

from dainemo import GRAPH
from dainemo.autograd.node import Node
from dainemo.utils.tensorutils import dot, tsum, elwise_op, elwise_pow, elwise_transform, fill, batch_tensor_elwise_op

from math import add, sub, mul, div, log


'''
Implement forward and backward operations for basic tensor manipulations.
'''

# <------------ADD------------>
struct ADD:
    @staticmethod
    fn forward(n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of element wise addition.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype]
        if n1.tensor.shape() == n2.tensor.shape():
            res = elwise_op[dtype, nelts, add](n1.tensor, n2.tensor)
        else:
            res = batch_tensor_elwise_op[dtype, nelts, add](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn backward(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise addition.'''
        # d(x + y) / dx = d(x + y) / dy = 1
        return ug


# <------------SUB------------>
struct SUB:
    @staticmethod
    fn forward(n1: Node[dtype], n2: Node[dtype]) -> Node[dtype]:
        '''Forward operation of element wise subtraction.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] 
        if n1.tensor.shape() == n2.tensor.shape():
            res = elwise_op[dtype, nelts, sub](n1.tensor, n2.tensor)
        else:
            res = batch_tensor_elwise_op[dtype, nelts, sub](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn backward(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise subtraction.'''
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
        '''Forward operation of element wise multiplication.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype]
        if n1.tensor.shape() == n2.tensor.shape():
            res = elwise_op[dtype, nelts, mul](n1.tensor, n2.tensor)
        else:
            res = batch_tensor_elwise_op[dtype, nelts, mul](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn forward(n1: Node[dtype], a: SIMD[dtype, 1]) -> Node[dtype]:
        '''Forward operation of tensor-scalar multiplication.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_op[dtype, nelts, mul](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return GRAPH.create_graph_node[Self.backward](res, n1, Node[dtype](a_tensor))

    @staticmethod
    fn backward(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise multiplication.'''
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
            res = batch_tensor_elwise_op[dtype, nelts, div](n1.tensor, n2.tensor)
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
        '''Forward operation of dot product.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = dot[dtype, nelts](n1.tensor, n2.tensor)
        return GRAPH.create_graph_node[Self.backward](res, n1, n2)

    @staticmethod
    fn backward(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''Backward operation of dot product.'''
        # TODO: Only 2D input tensors are supported yet !! 
        from dainemo.utils.tensorutils import transpose_2D
        alias nelts: Int = simdwidthof[dtype]()
        if tensor_id == 0:
            let n2 = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])]
            return dot[dtype, nelts](ug, transpose_2D[dtype, nelts](n2.tensor))           # dot(ug, n2.T)
        else:
            let n1 = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])]
            return dot[dtype, nelts](transpose_2D[dtype, nelts](n1.tensor), ug)           # dot(n1.T, ug)




# <------------EXP------------>
# TODO


# <------------LOG------------>
# TODO


# <------------POW------------>
struct POW:
    @staticmethod
    fn forward(n1: Node[dtype], a: Int) -> Node[dtype]:
        '''Forward operation of element wise pow.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = elwise_pow[dtype, nelts](n1.tensor, a)
        var a_tensor: Tensor[dtype] = Tensor[dtype](1)
        a_tensor[0] = a
        return GRAPH.create_graph_node[Self.backward](res, n1, Node[dtype](a_tensor))

    @staticmethod
    fn backward(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''Backward operation of element wise pow.'''
        # By design: tensor has id = 0 and scalar has id 1
        alias nelts: Int = simdwidthof[dtype]()
        let a: SIMD[dtype, 1] = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[1])].tensor[0]
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor

        if tensor_id == 0:
            # d(x^y) / dx = y * x^(y-1)
            let res = elwise_op[dtype, nelts, mul](a, elwise_pow[dtype, nelts](t, a.to_int() - 1))      # a * t^(a-1)
            return elwise_op[dtype, nelts, mul](res, ug)                                                # a * t^(a-1) * ug
        else:
            # d(x^y) / dy = x^y * log(x)
            let t_a = elwise_pow[dtype, nelts](t, a.to_int())                           # t^a
            let log_t = elwise_transform[dtype, nelts, log](t)                          # log(t)
            let res = elwise_op[dtype, nelts, mul](t_a, log_t)                          # t^a * log(t)
            return elwise_op[dtype, nelts, mul](res, ug)                                # t^a * log(t) * ug


# <------------SUM------------>
struct SUM:
    @staticmethod
    fn forward[axis: Int](n: Node[dtype]) -> Node[dtype]:
        '''Forward pass of sum operation: along axis.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: Tensor[dtype] = tsum[dtype, nelts](n.tensor, axis=axis)
        return GRAPH.create_graph_node[Self.backward[axis=axis]](res, n)

    @staticmethod
    fn forward(n: Node[dtype]) -> Node[dtype]:
        '''Forward pass of sum operation: all elements.'''
        alias nelts: Int = simdwidthof[dtype]()
        let res: SIMD[dtype, 1] = tsum[dtype, nelts](n.tensor)
        var res_tensor = Tensor[dtype](1)
        res_tensor[0] = res
        return GRAPH.create_graph_node[Self.backward[axis=-1]](res_tensor, n)

    @staticmethod
    fn backward[axis: Int = -1](ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''Backward pass of sum operation.'''
        # TODO: Only 2D input tensors are supported yet !! 
        # By design only one node in the collection
        # Output tensor has always the same shape as node input tensor
        alias nelts: Int = simdwidthof[dtype]()
        let t = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor
        var res = Tensor[dtype](t.shape())
        fill[dtype, nelts](res, 1.0)
        
        if axis == -1:
            # Upper gradient will be a Tensor of shape: 1 scalar, as it was constructed by summing all elements of node.tensor
            return elwise_op[dtype, nelts, mul](res, ug[0])

        elif axis == 0:
            # Upper gradient will be a Tensor of shape: sum of node.tensor along axis 0
            return batch_tensor_elwise_op[dtype, nelts, mul](res, ug)

        elif axis == 1:
            # Upper gradient will be a Tensor of shape: sum of node.tensor along axis 1
            # TODO: Workaround since batch_tensor_elwise_op is only implemented across axis = 0
            from dainemo.utils.tensorutils import transpose_2D
            return transpose_2D[dtype, nelts](batch_tensor_elwise_op[dtype, nelts, mul](transpose_2D[dtype, nelts](res), transpose_2D[dtype, nelts](ug)))

        else:
            print("NotImplemented: Tensor Sum only support up to rank 2.")
            return res


# <---------TRANSPOSE--------->
# TODO


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
    fn backward(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''
        Reshape upper gradient to original shape.
        '''
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
    fn backward(ug: Tensor[dtype], tensor_vec: DynamicVector[String], tensor_id: Int) -> Tensor[dtype]:
        '''
        Reshape upper gradient to original shape.
        '''
        var res = ug
        let shape = GRAPH.graph[GRAPH.get_node_idx(tensor_vec[0])].tensor.shape()
        
        try:
            res.ireshape(shape)
        except:
            print("[ERROR]: Cannot reshape tensor in reshape backward pass.")

        return res
