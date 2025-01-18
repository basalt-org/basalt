from python.python import Python
from collections.optional import Optional, OptionalReg

from .node import Node
from .attributes import AttributeVector, Attribute
from .symbol import Symbol
from .ops import OP, static_result_shape, dynamic_result_shape
from .params import ParamDict, Param

from basalt import seed, dtype
from basalt import Tensor, TensorShape


struct Graph:
    var inputs: List[Symbol]
    var params: ParamDict
    var nodes: List[Node]
    var outputs: List[Symbol]
    var loss_out: OptionalReg[Symbol]
    var symbol_count: UInt32

    fn __init__(inout self):
        self.inputs = List[Symbol]()
        self.params = ParamDict()
        self.nodes = List[Node]()
        self.outputs = List[Symbol]()
        self.loss_out = None
        self.symbol_count = 0

    fn __moveinit__(inout self, owned other: Graph):
        self.inputs = other.inputs^
        self.params = other.params^
        self.nodes = other.nodes^
        self.outputs = other.outputs^
        self.loss_out = other.loss_out
        self.symbol_count = other.symbol_count

    fn create_symbol(inout self, shape: TensorShape, data: Optional[Param] = None, trainable: Bool = False, is_input: Bool = False) -> Symbol:
        var symbol = Symbol(self.symbol_count, dtype, shape, trainable)
        self.symbol_count += 1

        if is_input:
            self.inputs.append(symbol)
        else:
            if data is not None:
                self.params.put(symbol, data.value())
            else:
                self.params.put(symbol)

        return symbol

    fn input(inout self, shape: TensorShape, trainable: Bool = False) -> Symbol:
        return self.create_symbol(shape, trainable=trainable, is_input=True)

    fn param(inout self, shape: TensorShape, init: Param, trainable: Bool = True) -> Symbol:
        return self.create_symbol(shape, init, trainable)

    fn param(inout self, shape: TensorShape, trainable: Bool = True) -> Symbol:
        return self.create_symbol(shape, trainable=trainable)

    fn scalar(inout self, value: Scalar[dtype]) -> Symbol:
        return self.create_symbol(TensorShape(1), Param(value), trainable=False)

    fn constant(inout self, shape: TensorShape, data: List[Scalar[dtype]]) -> Symbol:
        return self.create_symbol(shape, Param(data), trainable=False)

    fn out(inout self, symbol: Symbol):
        self.outputs.append(symbol)

    fn loss(inout self, symbol: Symbol):
        self.loss_out = symbol

    fn op(
        inout self,
        op: OP,
        *operands: Symbol,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        var res_shape = static_result_shape(op, operands, attributes)
        var res = Symbol(self.symbol_count, dtype, res_shape, self.result_trainable(operands))
        self.symbol_count += 1

        var inputs = List[Symbol]()
        inputs.reserve(len(operands))

        for operand in operands:
            inputs.append(operand)

        self.nodes.append(Node(op, inputs, List[Symbol](res), attributes))
        return res

    fn op(
        inout self,
        op: OP,
        operand_1: Symbol,
        operand_2: Float64,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        return self.op(op, operand_1, self.scalar(operand_2.cast[dtype]()), attributes=attributes)

    fn op(
        inout self,
        op: OP,
        operand_1: Float64,
        operand_2: Symbol,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        return self.op(op, self.scalar(operand_1.cast[dtype]()), operand_2, attributes=attributes)

    fn create_symbols(inout self, shapes: List[TensorShape], trainable: Bool = False) -> List[Symbol]:
        var symbols = List[Symbol]()
        symbols.reserve(len(shapes))
        
        for shape in shapes:
            symbols.append(Symbol(self.symbol_count, dtype, shape[], trainable))
            self.symbol_count += 1
        
        return symbols

    fn add_node(inout self, op: OP, inputs: List[Symbol], outputs: List[Symbol], attributes: AttributeVector):
        self.nodes.append(Node(op, inputs, outputs, attributes))

    fn concat(inout self, *operands: Symbol, dim: Int = 0) -> Symbol:
        var attributes = AttributeVector(Attribute("dim", dim))
        var res_shape = dynamic_result_shape(OP.CONCAT, operands, attributes)[0]
        var res_symbols = self.create_symbols(List[TensorShape](res_shape), self.result_trainable(operands))

        var operand_list = List[Symbol]()
        operand_list.reserve(len(operands))
        for operand in operands:
            operand_list.append(operand)

        self.add_node(OP.CONCAT, operand_list, res_symbols, attributes)
        return res_symbols[0]

    fn split(
        inout self, operand: Symbol, sections: List[Int], dim: Int = 0
    ) -> List[Symbol]:
        var attributes = AttributeVector(Attribute("sections", TensorShape(sections)), Attribute("dim", dim))
        var res_shapes = dynamic_result_shape(OP.SPLIT, operand, attributes)
        var trainable = self.result_trainable(operand)
        var result_symbols = self.create_symbols(res_shapes, trainable)
        self.add_node(OP.SPLIT, List[Symbol](operand), result_symbols, attributes)
        return result_symbols

    @staticmethod
    fn result_trainable(operands: VariadicList[Symbol]) -> Bool:
        for operand in operands:
            if operand.trainable:
                return True
        return False

    fn json(self) -> String:
        var result: String = '{"graph_name": "basalt", "nodes": ['
        for i in range(len(self.nodes)):
            result += self.nodes[i].json()
            if i < len(self.nodes) - 1:
                result += ", "
        result += '], "inputs": ['
        for i in range(len(self.inputs)):
            result += self.inputs[i].json()
            if i < len(self.inputs) - 1:
                result += ", "
        result += '], "outputs": ['
        for i in range(len(self.outputs)):
            result += self.outputs[i].json()
            if i < len(self.outputs) - 1:
                result += ", "
        if self.loss_out:
            result += '], "loss": ['
            result += self.loss_out.value().json()
        result += '], "params": ['
        for i in range(len(self.params)):
            result += self.params.symbols[i].json()
            if i < len(self.params) - 1:
                result += ", "
        result += "]}"
        return result

    fn render(self, render_type: String = "node") raises:
        Python.add_to_path("./basalt/utils")
        var renderer = Python.import_module("graph_render")
        var json = Python.import_module("json")
        _ = renderer.netron_render(json.loads(self.json()), render_type)

    fn compile(inout self):
        # 0. Sorting the graph
        # The staticlly defined graph has an implicit topological sorted order because,
        # each new operation is added the list of nodes after its dependencies have been calculated.
        # This eliminates the need for explicit topological sorting.

        # Possibilities:
        # - 1. Graph layout transformation (graph rewrite)
        #       - Layer pruning (removing nodes that have no effect - with common sub-tree identification)
        #       - Eliminate redundant intermediate data copies
        #       - Operator replacement (e.g. replacing (combination of) costly ops with more efficient ones)
        #       - (exmple of graph rewrite: https://dl.acm.org/doi/pdf/10.1145/3453483.3454083  -  Table 4)
        #       - Other intra-block optimizations: (e.g. data layout transformation BCHW -> BHWC, etc.)
        # - 2. Operator fusion (combining ops without materializing intermediate results)
        #       - Fusion plan exploration
        #       - Fusion plan generation (with subsequent intra-block optimizations)
        #       - (example fusion plan algorithm: https://dl.acm.org/doi/pdf/10.1145/3453483.3454083   -   Listing 1)
        # - 3. Fusion Code generation (behaviour)
        #       - Code generation for planned fusion blocks
        #       - Other inter-block optimizations (e.g. data layout transformation BCHW -> BHWC, etc.)
        # - 4. Auto-tuning (of vectorization-, parallelization-, tiling-, unrolling-parameters)
        #       - (Might only work when memory is initialized)

        # Other considerations:
        # - Efficient Memory management:
        #       - Memory reuse (in-place operations)
        #       - Data layout from BCHW (batch, channel, height, width) to BHWC can lead to better utilization and efficiency
        # - VJP, JVP (for automatic differentiation)

        pass
