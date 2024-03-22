from collections.optional import Optional

from basalt import Graph, Symbol, Tensor, TensorShape
from basalt.autograd.ops import forward_op, backward_op
from basalt.utils.collection import Collection
from basalt.utils.tensorutils import fill
from .initializers import initialize_tensor


fn dv_contains(dv: DynamicVector[Symbol], symbol: Symbol) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False

# TODO: remove when ability to concatenate graphs
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


fn collect_trainable_parameters(g: Graph) -> DynamicVector[Symbol]:
    """
    Collect all symbols of trainable parameters.
    """

    var trainable_parameters = DynamicVector[Symbol]()
    
    for i in range(len(g.params)):
        if g.params.symbols[i].trainable:
            trainable_parameters.push_back(g.params.symbols[i])

    return trainable_parameters ^


struct Parameters[g: Graph]():
    var params: Collection
    var grads: Collection
    
    alias trainable_parameters = collect_trainable_parameters(g)

    fn __init__(inout self):
        
        # Max number of tensors to initialize (max capacity to avoid resizing)
        # Assumption: An input or a param cannot be an output of a node
        # Assumption: There is only one output tensor per node
        var N = len(g.inputs) + len(g.params) + len(g.nodes)
        self.params = Collection(capacity=N)
        self.grads = Collection(capacity=N)


struct Model[
    g: Graph,
    n_inference_nodes: Optional[Int] = calc_n_inference_nodes(g) # TODO: remove this
]():
    
    var parameters: Parameters[g]

    fn __init__(inout self, inference_only: Bool = False):

        self.parameters = Parameters[g]()
        self.allocate_tensor_memory()
        self.allocate_grad_memory()
        
        # TODO: ability to concatenate graphs
        # NOTE: inference_only only used for surpressing the warning.
        if not inference_only and not g.loss_out:
            print("\n\n[WARNING]: No loss defined, model.forward() unavailable!\n\n")
        if not n_inference_nodes:
            print("\n\n[WARNING]: No graph out defined, model.inference() unavailable!\n\n")


    # TODO: ability to concatenate graphs
    # Removes the need for splitting in forward and inference mode
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
        # TODO: known copy (create reference)
        return self.parameters.params[g.loss_out.value()]


    fn inference(inout self, *t_inputs: Tensor[dtype]) -> DynamicVector[Tensor[dtype]]:
        
        # 1. Execute forward pass up to model out
        self.execute[n_inference_nodes.value()](t_inputs)
        
        # 2. Return outputs from allocated output memory
        # TODO: known copies (create reference)
        var outputs = DynamicVector[Tensor[dtype]]()
        for i in range(len(g.outputs)):
            outputs.push_back(self.parameters.params[g.outputs[i]])

        return outputs ^


    fn execute[num_nodes: Int](inout self, t_input: VariadicListMem[Tensor[dtype]]):
        # 1. Write inputs to allocated input memory
        for i in range(len(g.inputs)):
            self.parameters.params[g.inputs[i]] = t_input[i]

        # 2. Loop over all nodes and execute forward operations        
        @parameter
        fn fw_unroll[i: Int]():
            alias op = g.nodes[i].operator
            alias t1 = g.nodes[i].input_1
            alias out = g.nodes[i].output
            alias attrs = g.nodes[i].attributes

            @parameter
            if op.num_operands == 1:
                # Unary operator
                forward_op[op, t1.shape, attrs](
                    self.parameters.params[out],
                    self.parameters.params[t1]
                )
            elif op.num_operands == 2:
                # Binary operator
                alias t2 = g.nodes[i].input_2.value()
                forward_op[op, t1.shape, t2.shape, attrs](
                    self.parameters.params[out],
                    self.parameters.params[t1],
                    self.parameters.params[t2]
                )
            elif op.num_operands == 3:
                # Ternary operator
                alias t2 = g.nodes[i].input_2.value()
                alias t3 = g.nodes[i].input_3.value()
                forward_op[op, t1.shape, t2.shape, t3.shape, attrs](
                    self.parameters.params[out],
                    self.parameters.params[t1],
                    self.parameters.params[t2],
                    self.parameters.params[t3]
                )

        unroll[fw_unroll, num_nodes]()


    fn backward(inout self):
        """
        Main entrypoint of backward pass.
        """

        # 1. Initialize output gradient at the beginning of the backward pass
        fill(self.parameters.grads[g.loss_out.value()], 1.0)

        # 2. Loop over all nodes in reverse order and execute backward operations
        @parameter
        fn bw_unroll[i: Int]():
            alias reverse_i = g.nodes.size - i - 1
            alias op = g.nodes[reverse_i].operator
            alias out = g.nodes[reverse_i].output  # or upper_grad symbol
            alias t1 = g.nodes[reverse_i].input_1
            alias attrs = g.nodes[reverse_i].attributes

            @parameter
            if op.num_operands == 1:
                # Unary operator
                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.grads[t1],     # grad to be updated: input_1
                    )
            
            elif op.num_operands == 2:
                # Binary operator
                alias t2 = g.nodes[reverse_i].input_2.value()
                
                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, t2.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.grads[t1],     # grad to be updated: input_1
                    )
                
                @parameter
                if t2.trainable:
                    backward_op[1, op, out.shape, t1.shape, t2.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.grads[t2],     # grad to be updated: input_2
                    )
            
            elif op.num_operands == 3:
                # Ternary operator
                alias t2 = g.nodes[reverse_i].input_2.value()
                alias t3 = g.nodes[reverse_i].input_3.value()

                @parameter
                if t1.trainable:
                    backward_op[0, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.params[t3],
                        self.parameters.grads[t1],     # grad to be updated: input_1
                    )

                @parameter
                if t2.trainable:
                    backward_op[1, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.params[t3],
                        self.parameters.grads[t2],     # grad to be updated: input_2
                    )

                @parameter
                if t3.trainable:
                    backward_op[2, op, out.shape, t1.shape, t2.shape, t3.shape, attrs](
                        self.parameters.grads[out],
                        self.parameters.params[t1],
                        self.parameters.params[t2],
                        self.parameters.params[t3],
                        self.parameters.grads[t3],     # grad to be updated: input_3
                    )

        unroll[bw_unroll, g.nodes.size]()


    fn allocate_tensor_memory(inout self):
        for i in range(len(g.inputs)):
            self.parameters.params.append(Tensor[dtype](g.inputs[i].shape), g.inputs[i])

        for i in range(len(g.params)):
            # Parameter initialization
            var par: Tensor[dtype]
            if g.params.values[i].initializer:
                # 1. Specific parameter initialization defined
                var initializer_attr = g.params.values[i].initializer.value()
                par = initialize_tensor(
                    shape=g.params.symbols[i].shape,
                    type=initializer_attr.value.to_string(),
                    data=g.params.values[i].data.value()
                )
            elif g.params.values[i].data:
                # 2. Parameter initialized with data only
                # Data is assumed to contain the tensor
                par = g.params.get_tensor(i)
            else:
                # Default parameter initialization to zero
                par = Tensor[dtype](g.params.symbols[i].shape)
            
            self.parameters.params.append(par, g.params.symbols[i])
        
        for i in range(len(g.nodes)):
            # Assumption: There is only one output tensor per node
            # Assumption: An input or a param cannot be an output of a node
            self.parameters.params.append(Tensor[dtype](g.nodes[i].output.shape), g.nodes[i].output)
    

    fn allocate_grad_memory(inout self):
        # Inputs don't have gradients.
        # Gradient have same shape as the tensor
        for i in range(len(g.params)):
            var grad = g.params.symbols[i]
            if grad.trainable:
                self.parameters.grads.append(Tensor[dtype](grad.shape), grad)

        for i in range(len(g.nodes)):
            var out = g.nodes[i].output
            if out.trainable:
                self.parameters.grads.append(Tensor[dtype](out.shape), out)