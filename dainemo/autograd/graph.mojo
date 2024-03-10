from python.python import Python
from tensor import TensorShape
from collections.optional import Optional

from .node import Node
from .attributes import AttributeVector, Attribute
from .symbol import Symbol
from .ops import OP, static_result_shape
from .constant import Constant, ConstantDict

from dainemo import seed, dtype
from dainemo.utils.uuid import UUID, ID


@value
struct Graph:
    var uuid: UUID
    var inputs: DynamicVector[Symbol]
    var output: Symbol
    var params: DynamicVector[Symbol]
    var nodes: DynamicVector[Node]

    var constants: ConstantDict[dtype]

    fn __init__(inout self):
        self.uuid = UUID(seed)
        self.inputs = DynamicVector[Symbol]()
        self.params = DynamicVector[Symbol]()
        self.nodes = DynamicVector[Node]()
        self.output = Symbol(ID(), dtype, TensorShape(-1), False)

        self.constants = ConstantDict[dtype]()

    fn input(inout self, shape: TensorShape) -> Symbol:
        var inp = Symbol(self.uuid.next(), dtype, shape, False)
        self.inputs.push_back(inp)
        return inp

    fn param(inout self, shape: TensorShape, requires_grad: Bool = True) -> Symbol:
        var par = Symbol(self.uuid.next(), dtype, shape, requires_grad)
        self.params.push_back(par)
        return par

    fn scalar(inout self, value: SIMD[dtype, 1]) -> Symbol:
        var cst = Constant(value)
        var scalar_id = Symbol(self.uuid.next(), cst.rank, dtype, cst.static_shape, requires_grad=False)

        # self.params.push_back(scalar_id)
        self.constants.put(scalar_id, cst)

        return scalar_id

    fn out(inout self, symbol: Symbol):
        self.output = symbol

    fn op(inout self, op: OP,
        operand_1: Symbol,
        operand_2: Optional[Symbol] = None,
        operand_3: Optional[Symbol] = None,
        attributes: AttributeVector = AttributeVector()
    ) -> Symbol:
        
        var res: Symbol
        if operand_3:
            res = Symbol(
                self.uuid.next(),
                dtype,
                static_result_shape(op, operand_1.shape(), operand_2.value().shape(), operand_3.value().shape(), attributes),
                self.result_requires_grad(operand_1, operand_2.value(), operand_3.value()),
            )
        elif operand_2:
            res = Symbol(
                self.uuid.next(),
                dtype,
                static_result_shape(op, operand_1.shape(), operand_2.value().shape(), attributes),
                self.result_requires_grad(operand_1, operand_2.value()),
            )
        else:
            res = Symbol(
                self.uuid.next(),
                dtype,
                static_result_shape(op, operand_1.shape(), attributes),
                self.result_requires_grad(operand_1),
            )

        self.nodes.push_back(Node(op, res, operand_1, operand_2.take(), operand_3.take(), attributes))
        return res ^

    fn op(inout self, op: OP,
        operand_1: Symbol,
        operand_2: FloatLiteral,
        attributes: AttributeVector = AttributeVector()
    ) -> Symbol:
        
        var operand_2_symbol = self.scalar(operand_2)
        var res = Symbol(
            self.uuid.next(),
            dtype,
            static_result_shape(op, operand_1.shape(), operand_2_symbol.shape(), attributes),
            self.result_requires_grad(operand_1),
        )

        self.nodes.push_back(Node(op, res, operand_1, operand_2_symbol, attributes=attributes))
        return res ^

    fn op(inout self, op: OP,
        operand_1: FloatLiteral,
        operand_2: Symbol,
        attributes: AttributeVector = AttributeVector()
    ) -> Symbol:
        
        var operand_1_symbol = self.scalar(operand_1)
        var res = Symbol(
            self.uuid.next(),
            dtype,
            static_result_shape(op, operand_1_symbol.shape(), operand_2.shape(), attributes),
            self.result_requires_grad(operand_2),
        )

        self.nodes.push_back(Node(op, res, operand_1_symbol, operand_2, attributes=attributes))
        return res ^

    @staticmethod
    fn result_requires_grad(operand_1: Symbol) -> Bool:
        return operand_1.requires_grad

    @staticmethod
    fn result_requires_grad(operand_1: Symbol, operand_2: Symbol) -> Bool:
        return operand_1.requires_grad or operand_2.requires_grad

    @staticmethod
    fn result_requires_grad(operand_1: Symbol, operand_2: Symbol, operand_3: Symbol) -> Bool:
        return operand_1.requires_grad or operand_2.requires_grad or operand_3.requires_grad

    fn json(self) -> String:
        var result: String = '{"graph_name": "Dainemo", "nodes": ['
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
        result += self.output.json()
        result += '], "params": ['
        for i in range(len(self.constants)):
            result += self.constants.keys[i].json()
            result += ", "
        for i in range(len(self.params)):
            result += self.params[i].json()
            if i < len(self.params) - 1:
                result += ", "
        result += "]}"
        return result

    fn render(self, render_type: String = "node") raises:
        Python.add_to_path("./dainemo/utils")
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
