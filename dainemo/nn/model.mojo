from math import sqrt

from dainemo import Graph, Symbol
from dainemo.autograd.ops import forward_op, backward_op, OP
from dainemo.utils.collection import Collection
from dainemo.utils.tensorutils import fill
from dainemo.utils.rand_utils import rand_uniform
from dainemo.utils.string_dict import StringDict



fn dv_contains(dv: DynamicVector[Symbol], symbol: Symbol) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False


fn calc_n_tensors(g: Graph) -> Int:
    var num: Int = len(g.inputs) + len(g.params)
    var visited_results = DynamicVector[Symbol]()
    for i in range(len(g.nodes)):
        if not dv_contains(visited_results, g.nodes[i].output):
            visited_results.push_back(g.nodes[i].output)
            num += 1
    return num


struct Parameters():
    var params: Collection # capacity = n_tensors = n
    var grads: Collection  # n_grads < n_tensors = N (as some require_grad = False)
    var params_map: StringDict[Int]
    var grads_map: StringDict[Int]

    fn __init__(inout self, N: Int):
        self.params = Collection(N)
        self.grads = Collection(N)
        self.params_map = StringDict[Int]()
        self.grads_map = StringDict[Int]()


struct Model[
    g: Graph,
    N: Int = calc_n_tensors(g)
]():

    var parameters: Parameters

    fn __init__(inout self):
        self.parameters = Parameters(N)

        self.allocate_tensor_memory()
        self.allocate_grad_memory()


    fn forward(inout self, owned *t_input: Tensor[dtype]) -> Tensor[dtype]:
        # 1. Write inputs to allocated input memory
        for i in range(len(g.inputs)):
            var input_idx = self.parameters.params_map.get(str(g.inputs[i].name), -1)
            __get_address_as_lvalue(self.parameters.params.offset(input_idx).address) = t_input[i]

        # 2. Loop over all nodes and execute forward operations
        var res_idx: Int = 0
        var in1_idx: Int = 0
        var in2_idx: Int = 0
        
        @parameter
        fn fw_unroll[i: Int]():
            res_idx = self.parameters.params_map.get(str(g.nodes[i].output.name), -1)
            in1_idx = self.parameters.params_map.get(str(g.nodes[i].input_1.name), -1)
            in2_idx = self.parameters.params_map.get(str(g.nodes[i].input_2.value().name), -1)

            if in2_idx == -1:
                forward_op[g.nodes[i].operator, g.nodes[i].input_1.shape(), g.nodes[i].attributes](
                    __get_address_as_lvalue(self.parameters.params.offset(res_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in1_idx).address)
                )
            else:
                forward_op[g.nodes[i].operator, g.nodes[i].input_1.shape(), g.nodes[i].input_2.value().shape(), g.nodes[i].attributes](
                    __get_address_as_lvalue(self.parameters.params.offset(res_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in1_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in2_idx).address)
                )

        unroll[fw_unroll, g.nodes.size]()

        # 3. Return output from allocated output memory
        var out_idx: Int = 0
        out_idx = self.parameters.params_map.get(str(g.output.name), -1)

        return __get_address_as_lvalue(self.parameters.params.offset(out_idx).address)


    fn backward(inout self):
        """
        Main entrypoint of backward pass.
        """

        # 1. Initialize output gradient at the beginning of the backward pass
        var output_idx: Int = 0

        output_idx = self.parameters.grads_map.get(str(g.output.name), -1)
        fill[dtype, nelts](__get_address_as_lvalue(self.parameters.grads.offset(output_idx).address), 1.0)

        # 2. Loop over all nodes in reverse order and execute backward operations
        var grad_ug_idx: Int = 0
        var grad_in1_idx: Int = 0
        var grad_in2_idx: Int = 0
        var tensor_in1_idx: Int = 0
        var tensor_in2_idx: Int = 0

        @parameter
        fn bw_unroll[i: Int]():
            alias reverse_i = g.nodes.size - i - 1
            grad_ug_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].output.name), -1)
            grad_in1_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_1.name), -1)
            grad_in2_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_2.value().name), -1)
            tensor_in1_idx = self.parameters.params_map.get(str(g.nodes[reverse_i].input_1.name), -1)
            tensor_in2_idx = self.parameters.params_map.get(str(g.nodes[reverse_i].input_2.value().name), -1)

            # If input_1 or input_2 does not require gradient, grad_in1_idx or grad_in2_idx will be -1
            # Because they were not added to the keys of grad_map
            if tensor_in2_idx == -1:
                if grad_in1_idx != -1:
                    backward_op[ 
                        0,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),              # input_1 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in1_idx).address),     # grad to be updated: input_1
                    )
            else:
                if grad_in1_idx != -1:
                    backward_op[ 
                        0,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),             # input_1 shape
                        g.nodes[reverse_i].input_2.value().shape(),     # input_2 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in2_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in1_idx).address),     # grad to be updated: input_1
                    )
                if grad_in2_idx != -1:
                    backward_op[ 
                        1,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),             # input_1 shape
                        g.nodes[reverse_i].input_2.value().shape(),      # input_2 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in2_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in2_idx).address),     # grad to be updated: input_2
                    )

        unroll[bw_unroll, g.nodes.size]()


    fn allocate_tensor_memory(inout self):
        for i in range(len(g.inputs)):
            self.parameters.params_map.put(str(g.inputs[i].name), self.parameters.params.size)
            self.parameters.params.append(Tensor[dtype](g.inputs[i].shape()))

        for i in range(len(g.params)):
            self.parameters.params_map.put(str(g.params[i].name), self.parameters.params.size)
            var par = Tensor[dtype](g.params[i].shape())
            
            # Parameter initialization
            var k: SIMD[dtype, 1] = 1.0 / par.dim(0)
            rand_uniform(par, -sqrt(k), sqrt(k))
            self.parameters.params.append(par)

        for i in range(len(g.constants.keys)):
            self.parameters.params_map.put(str(g.constants.keys[i].name), self.parameters.params.size)
            var cst = g.constants.get(g.constants.keys[i])
            self.parameters.params.append(cst)
        
        for i in range(len(g.nodes)):
            if not self.parameters.params_map.__contains__(str(g.nodes[i].output.name)):
                self.parameters.params_map.put(str(g.nodes[i].output.name), self.parameters.params.size)
                self.parameters.params.append(Tensor[dtype](g.nodes[i].output.shape()))
    

    fn allocate_grad_memory(inout self):
        # Inputs don't have gradients.
        # Gradient have same shape as the tensor
        for i in range(len(g.params)):
            if g.params[i].requires_grad:
                self.parameters.grads_map.put(str(g.params[i].name), self.parameters.grads.size)
                self.parameters.grads.append(Tensor[dtype](g.params[i].shape()))

        for i in range(len(g.nodes)):
            if not self.parameters.grads_map.__contains__(str(g.nodes[i].output.name)):
                if g.nodes[i].output.requires_grad:
                    self.parameters.grads_map.put(str(g.nodes[i].output.name), self.parameters.grads.size)
                    self.parameters.grads.append(Tensor[dtype](g.nodes[i].output.shape()))

    fn get_grad(inout self, id: String) -> Tensor[dtype]:
        var index = self.parameters.grads_map.get(id, -1)
        if index == -1:
            return Tensor[dtype]()

        return __get_address_as_lvalue(self.parameters.grads.offset(index).address)

    fn get_param(inout self, id: String) -> Tensor[dtype]:
        var index = self.parameters.params_map.get(id, -1)
        if index == -1:
            return Tensor[dtype]()

        return __get_address_as_lvalue(self.parameters.params.offset(index).address)