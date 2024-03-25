from python.python import Python
from collections.optional import Optional

from .node import Node
from .attributes import AttributeVector, Attribute
from .symbol import Symbol
from .ops import OP, static_result_shape
from .params import ParamDict, Param

from basalt import seed, dtype
from basalt import Tensor, TensorShape


@value
struct Graph:
    var inputs: DynamicVector[Symbol]
    var params: ParamDict
    var nodes: DynamicVector[Node]
    var outputs: DynamicVector[Symbol]
    var loss_out: Optional[Symbol]
    var symbol_count: UInt32

    fn __init__(inout self):
        self.inputs = DynamicVector[Symbol]()
        self.params = ParamDict()
        self.nodes = DynamicVector[Node]()
        self.outputs = DynamicVector[Symbol]()
        self.loss_out = None
        self.symbol_count = 0

    fn input(inout self, shape: TensorShape) -> Symbol:
        var inp = Symbol(self.symbol_count, dtype, shape, False)
        self.inputs.push_back(inp)
        self.symbol_count += 1
        return inp

    fn param(
        inout self, shape: TensorShape, init: Param, trainable: Bool = True
    ) -> Symbol:
        var param_id = Symbol(self.symbol_count, dtype, shape, trainable)
        self.params.put(param_id, init)
        self.symbol_count += 1
        return param_id

    fn param(inout self, shape: TensorShape, trainable: Bool = True) -> Symbol:
        var param_id = Symbol(self.symbol_count, dtype, shape, trainable)
        self.params.put(param_id)
        self.symbol_count += 1
        return param_id

    fn scalar(inout self, value: SIMD[dtype, 1]) -> Symbol:
        var scal = Param(value)
        var scalar_id = Symbol(
            self.symbol_count, dtype, TensorShape(1), trainable=False
        )
        self.params.put(scalar_id, scal)
        self.symbol_count += 1
        return scalar_id

    fn constant(
        inout self, shape: TensorShape, data: DynamicVector[SIMD[dtype, 1]]
    ) -> Symbol:
        var cst = Param(data)
        var constant_id = Symbol(self.symbol_count, dtype, shape, trainable=False)
        self.params.put(constant_id, cst)
        self.symbol_count += 1
        return constant_id

    fn out(inout self, symbol: Symbol):
        self.outputs.push_back(symbol)

    fn loss(inout self, symbol: Symbol):
        self.loss_out = symbol

    fn op(
        inout self,
        op: OP,
        operand_1: Symbol,
        operand_2: Optional[Symbol] = None,
        operand_3: Optional[Symbol] = None,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        var res: Symbol
        if operand_3:
            res = Symbol(
                self.symbol_count,
                dtype,
                static_result_shape(
                    op,
                    operand_1.shape,
                    operand_2.value().shape,
                    operand_3.value().shape,
                    attributes,
                ),
                self.result_trainable(operand_1, operand_2.value(), operand_3.value()),
            )
        elif operand_2:
            res = Symbol(
                self.symbol_count,
                dtype,
                static_result_shape(
                    op, operand_1.shape, operand_2.value().shape, attributes
                ),
                self.result_trainable(operand_1, operand_2.value()),
            )
        else:
            res = Symbol(
                self.symbol_count,
                dtype,
                static_result_shape(op, operand_1.shape, attributes),
                self.result_trainable(operand_1),
            )

        self.nodes.push_back(
            Node(op, res, operand_1, operand_2.take(), operand_3.take(), attributes)
        )
        self.symbol_count += 1
        return res

    fn op(
        inout self,
        op: OP,
        operand_1: Symbol,
        operand_2: FloatLiteral,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        var operand_2_symbol = self.scalar(operand_2)
        var res = Symbol(
            self.symbol_count,
            dtype,
            static_result_shape(
                op, operand_1.shape, operand_2_symbol.shape, attributes
            ),
            self.result_trainable(operand_1),
        )

        self.nodes.push_back(
            Node(op, res, operand_1, operand_2_symbol, attributes=attributes)
        )
        self.symbol_count += 1
        return res

    fn op(
        inout self,
        op: OP,
        operand_1: FloatLiteral,
        operand_2: Symbol,
        attributes: AttributeVector = AttributeVector(),
    ) -> Symbol:
        var operand_1_symbol = self.scalar(operand_1)
        var res = Symbol(
            self.symbol_count,
            dtype,
            static_result_shape(
                op, operand_1_symbol.shape, operand_2.shape, attributes
            ),
            self.result_trainable(operand_2),
        )

        self.nodes.push_back(
            Node(op, res, operand_1_symbol, operand_2, attributes=attributes)
        )
        self.symbol_count += 1
        return res

    @staticmethod
    fn result_trainable(operand_1: Symbol) -> Bool:
        return operand_1.trainable

    @staticmethod
    fn result_trainable(operand_1: Symbol, operand_2: Symbol) -> Bool:
        return operand_1.trainable or operand_2.trainable

    @staticmethod
    fn result_trainable(
        operand_1: Symbol, operand_2: Symbol, operand_3: Symbol
    ) -> Bool:
        return operand_1.trainable or operand_2.trainable or operand_3.trainable

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
