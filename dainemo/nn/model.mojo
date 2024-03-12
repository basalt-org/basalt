from math import sqrt
from collections.optional import Optional

from dainemo import Graph, Symbol
from dainemo.autograd.ops import forward_op, backward_op, OP
from dainemo.utils.collection import Collection
from dainemo.utils.tensorutils import fill
from dainemo.utils.rand_utils import rand_uniform
from dainemo.utils.string_dict import StringDict
from .initializers import initialize_tensor


fn dv_contains(dv: DynamicVector[Symbol], symbol: Symbol) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False


fn calc_n_tensors(g: Graph) -> Int:
    """
    Calculate the number of tensors required to store in a collection.
    """
    var num: Int = len(g.inputs) + len(g.params)
    var visited_results = DynamicVector[Symbol]()
    for i in range(len(g.nodes)):
        if not dv_contains(visited_results, g.nodes[i].output):
            visited_results.push_back(g.nodes[i].output)
            num += 1
    return num


fn calc_n_inference_nodes(g: Graph) -> Optional[Int]:
    """
    Calculate the index of the node up to wich the forward pass should be executed for a model inference.
    When looping in revers: Equals the first index on which the node output is also a graph output.
    The number of inference nodes is that index + 1.
    """
    for i in range(len(g.nodes) - 1, -1, -1):
        if dv_contains(g.outputs, g.nodes[i].output):
            return i + 1
    return None



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
    N: Int = calc_n_tensors(g),
    n_inference_nodes: Optional[Int] = calc_n_inference_nodes(g)
]():

    var parameters: Parameters

    fn __init__(inout self, inference_only: Bool = False):
        self.parameters = Parameters(N)

        self.allocate_tensor_memory()
        self.allocate_grad_memory()
        
        # NOTE: inference_only only used for surpressing the warning.
        if not inference_only and not g.loss_out:
            print("\n\n[WARNING]: No loss defined, model.forward() unavailable!\n\n")
        if not n_inference_nodes:
            print("\n\n[WARNING]: No graph out defined, model.inference() unavailable!\n\n")


    fn forward(inout self, *t_inputs: Tensor[dtype]) -> Tensor[dtype]:
        
        # NOTE: Important detail here is that the order of the inputs must be the same as the order the inputs were defined in the graph.
        # Example: If you were te define the y_true before the x when creating the graph
        #       
        #   var g = Graph()
        #   var y_true = g.input(TensorShape(batch_size, n_outputs))
        #   var x = g.input(TensorShape(batch_size, n_inputs))
        #
        # Then the order of the inputs in the forward call must be the same:
        #
        #   model.forward(batch.labels, batch.inputs)


        # 1. Execute a full forward pass (model inference + loss)
        self.execute[g.nodes.size](t_inputs)

        # 2. Return loss from allocated output memory
        var loss_out_idx: Int = self.parameters.params_map.get(str(g.loss_out.value().name), -1)

        return __get_address_as_lvalue(self.parameters.params.offset(loss_out_idx).address)


    fn inference(inout self, *t_inputs: Tensor[dtype]) -> DynamicVector[Tensor[dtype]]:
        
        # 1. Execute forward pass up to model out
        self.execute[n_inference_nodes.value()](t_inputs)
        
        # 2. Return output from allocated output memory
        var out_idx: Int
        var outputs = DynamicVector[Tensor[dtype]]()
        for i in range(len(g.outputs)):
            out_idx = self.parameters.params_map.get(str(g.outputs[i].name), -1)
            outputs.push_back(__get_address_as_lvalue(self.parameters.params.offset(out_idx).address))

        return outputs ^


    fn execute[num_nodes: Int](inout self, t_input: VariadicListMem[Tensor[dtype]]):
        # 1. Write inputs to allocated input memory
        for i in range(len(g.inputs)):
            var input_idx = self.parameters.params_map.get(str(g.inputs[i].name), -1)
            __get_address_as_lvalue(self.parameters.params.offset(input_idx).address) = t_input[i]

        # 2. Loop over all nodes and execute forward operations
        var res_idx: Int = 0
        var in1_idx: Int = 0
        var in2_idx: Int = 0
        var in3_idx: Int = 0
        
        @parameter
        fn fw_unroll[i: Int]():
            res_idx = self.parameters.params_map.get(str(g.nodes[i].output.name), -1)
            in1_idx = self.parameters.params_map.get(str(g.nodes[i].input_1.name), -1)

            @parameter
            if g.nodes[i].operator.num_operands == 1:
                # Unary operator
                forward_op[g.nodes[i].operator, g.nodes[i].input_1.shape(), g.nodes[i].attributes](
                    __get_address_as_lvalue(self.parameters.params.offset(res_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in1_idx).address)
                )
            elif g.nodes[i].operator.num_operands == 2:
                # Binary operator
                in2_idx = self.parameters.params_map.get(str(g.nodes[i].input_2.value().name), -1)

                forward_op[g.nodes[i].operator, g.nodes[i].input_1.shape(), g.nodes[i].input_2.value().shape(), g.nodes[i].attributes](
                    __get_address_as_lvalue(self.parameters.params.offset(res_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in1_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in2_idx).address)
                )
            elif g.nodes[i].operator.num_operands == 3:
                # Ternary operator
                in2_idx = self.parameters.params_map.get(str(g.nodes[i].input_2.value().name), -1)
                in3_idx = self.parameters.params_map.get(str(g.nodes[i].input_3.value().name), -1)

                forward_op[g.nodes[i].operator, g.nodes[i].input_1.shape(), g.nodes[i].input_2.value().shape(), g.nodes[i].input_3.value().shape(), g.nodes[i].attributes](
                    __get_address_as_lvalue(self.parameters.params.offset(res_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in1_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in2_idx).address),
                    __get_address_as_lvalue(self.parameters.params.offset(in3_idx).address)
                )

        unroll[fw_unroll, num_nodes]()


    fn backward(inout self):
        """
        Main entrypoint of backward pass.
        """

        # 1. Initialize output gradient at the beginning of the backward pass
        var output_idx: Int = 0

        output_idx = self.parameters.grads_map.get(str(g.loss_out.value().name), -1)
        fill[dtype, nelts](__get_address_as_lvalue(self.parameters.grads.offset(output_idx).address), 1.0)

        # 2. Loop over all nodes in reverse order and execute backward operations
        var grad_ug_idx: Int = 0
        var grad_in1_idx: Int = 0
        var grad_in2_idx: Int = 0
        var grad_in3_idx: Int = 0
        var tensor_in1_idx: Int = 0
        var tensor_in2_idx: Int = 0
        var tensor_in3_idx: Int = 0

        @parameter
        fn bw_unroll[i: Int]():
            alias reverse_i = g.nodes.size - i - 1
            grad_ug_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].output.name), -1)
            tensor_in1_idx = self.parameters.params_map.get(str(g.nodes[reverse_i].input_1.name), -1)
            
            @parameter
            if g.nodes[reverse_i].operator.num_operands == 1:
                # Unary operator
                @parameter
                if g.nodes[reverse_i].input_1.trainable:
                    grad_in1_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_1.name), -1)
                    backward_op[ 
                        0,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),             # input_1 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in1_idx).address),     # grad to be updated: input_1
                    )
            elif g.nodes[reverse_i].operator.num_operands == 2:
                # Binary operator
                tensor_in2_idx = self.parameters.params_map.get(str(g.nodes[reverse_i].input_2.value().name), -1)
                
                @parameter
                if g.nodes[reverse_i].input_1.trainable:
                    grad_in1_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_1.name), -1)
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
                
                @parameter
                if g.nodes[reverse_i].input_2.value().trainable:
                    grad_in2_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_2.value().name), -1)
                    backward_op[ 
                        1,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),             # input_1 shape
                        g.nodes[reverse_i].input_2.value().shape(),     # input_2 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in2_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in2_idx).address),     # grad to be updated: input_2
                    )
            
            elif g.nodes[reverse_i].operator.num_operands == 3:
                # Ternary operator
                tensor_in2_idx = self.parameters.params_map.get(str(g.nodes[reverse_i].input_2.value().name), -1)
                tensor_in3_idx = self.parameters.params_map.get(str(g.nodes[reverse_i].input_3.value().name), -1)

                @parameter
                if g.nodes[reverse_i].input_1.trainable:
                    grad_in1_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_1.name), -1)
                    backward_op[ 
                        0,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),             # input_1 shape
                        g.nodes[reverse_i].input_2.value().shape(),     # input_2 shape
                        g.nodes[reverse_i].input_3.value().shape(),     # input_3 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in2_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in3_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in1_idx).address),     # grad to be updated: input_1
                    )

                @parameter
                if g.nodes[reverse_i].input_2.value().trainable:
                    grad_in2_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_2.value().name), -1)
                    backward_op[ 
                        1,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),             # input_1 shape
                        g.nodes[reverse_i].input_2.value().shape(),     # input_2 shape
                        g.nodes[reverse_i].input_3.value().shape(),     # input_3 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in2_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in3_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in2_idx).address),     # grad to be updated: input_2
                    )

                @parameter
                if g.nodes[reverse_i].input_3.value().trainable:
                    grad_in3_idx = self.parameters.grads_map.get(str(g.nodes[reverse_i].input_3.value().name), -1)
                    backward_op[ 
                        2,
                        g.nodes[reverse_i].operator,
                        g.nodes[reverse_i].output.shape(),              # uppergrad shape
                        g.nodes[reverse_i].input_1.shape(),             # input_1 shape
                        g.nodes[reverse_i].input_2.value().shape(),     # input_2 shape
                        g.nodes[reverse_i].input_3.value().shape(),     # input_3 shape
                        g.nodes[reverse_i].attributes,
                    ](
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_ug_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in1_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in2_idx).address),
                        __get_address_as_lvalue(self.parameters.params.offset(tensor_in3_idx).address),
                        __get_address_as_lvalue(self.parameters.grads.offset(grad_in3_idx).address),     # grad to be updated: input_3
                    )

        unroll[bw_unroll, g.nodes.size]()


    fn allocate_tensor_memory(inout self):
        for i in range(len(g.inputs)):
            self.parameters.params_map.put(str(g.inputs[i].name), self.parameters.params.size)
            self.parameters.params.append(Tensor[dtype](g.inputs[i].shape()))

        for i in range(len(g.params)):
            self.parameters.params_map.put(str(g.params.symbols[i].name), self.parameters.params.size)
            
            var par: Tensor[dtype]
            if g.params.values[i].data:
                # Parameter initialized with data
                par = g.params.get_tensor(i)
            elif g.params.values[i].initializer:
                # Parameter initialization with attributes
                var initializer_attr = g.params.values[i].initializer.value()
                par = initialize_tensor(
                    shape=g.params.symbols[i].shape(),
                    type=initializer_attr.value.to_string()
                )
            else:
                # Default parameter initialization
                par = Tensor[dtype](g.params.symbols[i].shape())
            
            self.parameters.params.append(par)
        
        for i in range(len(g.nodes)):
            if not self.parameters.params_map.__contains__(str(g.nodes[i].output.name)):
                self.parameters.params_map.put(str(g.nodes[i].output.name), self.parameters.params.size)
                self.parameters.params.append(Tensor[dtype](g.nodes[i].output.shape()))
    

    fn allocate_grad_memory(inout self):
        # Inputs don't have gradients.
        # Gradient have same shape as the tensor
        for i in range(len(g.params)):
            if g.params.symbols[i].trainable:
                self.parameters.grads_map.put(str(g.params.symbols[i].name), self.parameters.grads.size)
                self.parameters.grads.append(Tensor[dtype](g.params.symbols[i].shape()))

        for i in range(len(g.nodes)):
            if not self.parameters.grads_map.__contains__(str(g.nodes[i].output.name)):
                if g.nodes[i].output.trainable:
                    self.parameters.grads_map.put(str(g.nodes[i].output.name), self.parameters.grads.size)
                    self.parameters.grads.append(Tensor[dtype](g.nodes[i].output.shape()))