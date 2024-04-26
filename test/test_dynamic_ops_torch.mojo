from random import rand
from python.python import Python
from test_tensorutils import assert_tensors_equal

import basalt.nn as nn
from basalt import Graph, Symbol, OP, GRADS
from basalt import Tensor, TensorShape
from basalt.autograd.attributes import Attribute, AttributeVector

from test_dynamic_ops import create_graph_concat, create_graph_split
from test_utils_extras import (
    to_numpy,
    to_tensor,
)

alias dtype = DType.float32
alias nelts: Int = simdwidthof[dtype]()


@value
struct torch_output_cat:
    var expected: Tensor[dtype]
    var grad_1: Tensor[dtype]
    var grad_2: Tensor[dtype]
    var grad_3: Tensor[dtype]


fn torch_cat(
    input_1: Tensor, input_2: Tensor, input_3: Tensor, upper_grad: Tensor, dim: Int
) -> torch_output_cat:
    try:
        var py = Python.import_module("builtins")
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")

        var input_1 = torch.from_numpy(to_numpy(input_1)).requires_grad_(True)
        var input_2 = torch.from_numpy(to_numpy(input_2)).requires_grad_(True)
        var input_3 = torch.from_numpy(to_numpy(input_3)).requires_grad_(True)

        var expected: PythonObject

        var tensors = py.list()
        tensors.append(input_1)
        tensors.append(input_2)
        tensors.append(input_3)
        expected = torch.cat(tensors, dim=dim)

        # uppergrad & backwards
        var upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        return torch_output_cat(
            to_tensor(expected.detach().numpy()),
            to_tensor(input_1.grad.numpy()),
            to_tensor(input_2.grad.numpy()),
            to_tensor(input_3.grad.numpy()),
        )

    except e:
        print("Error importing torch: ", e)
        var d = Tensor[dtype](1)
        return torch_output_cat(d, d, d, d)


fn test_CONCAT() raises:
    alias t1_shape = TensorShape(11, 3, 17, 19)
    alias t2_shape = TensorShape(11, 3, 17, 19)
    alias t3_shape = TensorShape(11, 3, 17, 19)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    var t3 = Tensor[dtype](t3_shape)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())
    rand(t3.data(), t3.num_elements())

    # default: dim = 0
    alias graph = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=0)
    var model = nn.Model[graph]()
    var res = model.forward(t1, t2, t3)

    alias ug_shape = TensorShape(33, 3, 17, 19)
    var ug = Tensor[dtype](ug_shape)
    rand(ug.data(), ug.num_elements())

    var expected_and_grad = torch_cat(t1, t2, t3, ug, dim=0)
    model.backward(ug)

    assert_tensors_equal(res, expected_and_grad.expected, "almost")
    assert_tensors_equal(
        GRADS[graph.nodes[0].inputs[0]], expected_and_grad.grad_1, "almost"
    )
    assert_tensors_equal(
        GRADS[graph.nodes[0].inputs[1]], expected_and_grad.grad_2, "almost"
    )
    assert_tensors_equal(
        GRADS[graph.nodes[0].inputs[2]], expected_and_grad.grad_3, "almost"
    )

    # dim = 2
    alias graph_2 = create_graph_concat(t1_shape, t2_shape, t3_shape, dim=2)
    var model_2 = nn.Model[graph_2]()
    var res_2 = model_2.forward(t1, t2, t3)

    alias ug_shape_2 = TensorShape(11, 3, 51, 19)
    var ug_2 = Tensor[dtype](ug_shape_2)
    rand(ug_2.data(), ug_2.num_elements())

    var expected_and_grad_2 = torch_cat(t1, t2, t3, ug_2, dim=2)
    model_2.backward(ug_2)

    assert_tensors_equal(res_2, expected_and_grad_2.expected, "almost")
    assert_tensors_equal(
        GRADS[graph_2.nodes[0].inputs[0]], expected_and_grad_2.grad_1, "almost"
    )
    assert_tensors_equal(
        GRADS[graph_2.nodes[0].inputs[1]], expected_and_grad_2.grad_2, "almost"
    )
    assert_tensors_equal(
        GRADS[graph_2.nodes[0].inputs[2]], expected_and_grad_2.grad_3, "almost"
    )


@value
struct torch_output_split:
    var expected1: Tensor[dtype]
    var expected2: Tensor[dtype]
    var expected3: Tensor[dtype]
    var grad: Tensor[dtype]


fn torch_split(
    input: Tensor,
    upper_grad_1: Tensor,
    upper_grad_2: Tensor,
    upper_grad_3: Tensor,
    sections: List[Int],
    dim: Int,
) -> torch_output_split:
    try:
        var py = Python.import_module("builtins")
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")

        var input = torch.from_numpy(to_numpy(input)).requires_grad_(True)

        var sizes = py.list()
        sizes.append(sections[0])
        sizes.append(sections[1])
        sizes.append(sections[2])

        var chunks: PythonObject = input.split(sizes, dim=dim)

        # uppergrad & backwards
        var upper_grad_1 = torch.from_numpy(to_numpy(upper_grad_1))
        var upper_grad_2 = torch.from_numpy(to_numpy(upper_grad_2))
        var upper_grad_3 = torch.from_numpy(to_numpy(upper_grad_3))
        _ = chunks[0].backward(upper_grad_1)
        _ = chunks[1].backward(upper_grad_2)
        _ = chunks[2].backward(upper_grad_3)

        return torch_output_split(
            to_tensor(chunks[0].detach().numpy()),
            to_tensor(chunks[1].detach().numpy()),
            to_tensor(chunks[2].detach().numpy()),
            to_tensor(input.grad.numpy()),
        )

    except e:
        print("Error importing torch: ", e)
        var d = Tensor[dtype](1)
        return torch_output_split(d, d, d, d)


fn test_SPLIT() raises:
    alias t1_shape = TensorShape(11, 3, 17, 19)
    var t1 = Tensor[dtype](t1_shape)
    rand(t1.data(), t1.num_elements())

    # default: dim = 0
    alias sections = List[Int](3, 6, 2)  # 11
    alias graph = create_graph_split(t1_shape, sections, dim=0)
    var model = nn.Model[graph]()
    var results = model.inference(t1)

    alias ug1_shape = TensorShape(3, 3, 17, 19)
    alias ug2_shape = TensorShape(6, 3, 17, 19)
    alias ug3_shape = TensorShape(2, 3, 17, 19)
    var ug1 = Tensor[dtype](ug1_shape)
    var ug2 = Tensor[dtype](ug2_shape)
    var ug3 = Tensor[dtype](ug3_shape)
    rand(ug1.data(), ug1.num_elements())
    rand(ug2.data(), ug2.num_elements())
    rand(ug3.data(), ug3.num_elements())

    var expected_and_grad = torch_split(t1, ug1, ug2, ug3, sections, dim=0)
    model.backward(ug1, ug2, ug3)

    assert_tensors_equal(results[0], expected_and_grad.expected1, "almost")
    assert_tensors_equal(results[1], expected_and_grad.expected2, "almost")
    assert_tensors_equal(results[2], expected_and_grad.expected3, "almost")
    assert_tensors_equal(
        GRADS[graph.nodes[0].inputs[0]], expected_and_grad.grad, "almost"
    )

    # dim = 2
    alias sections_2 = List[Int](3, 6, 8)  # 17
    alias graph_2 = create_graph_split(t1_shape, sections_2, dim=2)
    var model_2 = nn.Model[graph_2]()
    var results_2 = model_2.inference(t1)

    alias ug1_shape_2 = TensorShape(11, 3, 3, 19)
    alias ug2_shape_2 = TensorShape(11, 3, 6, 19)
    alias ug3_shape_2 = TensorShape(11, 3, 8, 19)
    var ug1_2 = Tensor[dtype](ug1_shape_2)
    var ug2_2 = Tensor[dtype](ug2_shape_2)
    var ug3_2 = Tensor[dtype](ug3_shape_2)
    rand(ug1_2.data(), ug1_2.num_elements())
    rand(ug2_2.data(), ug2_2.num_elements())
    rand(ug3_2.data(), ug3_2.num_elements())

    var expected_and_grad_2 = torch_split(t1, ug1_2, ug2_2, ug3_2, sections_2, dim=2)
    model_2.backward(ug1_2, ug2_2, ug3_2)

    assert_tensors_equal(results_2[0], expected_and_grad_2.expected1, "almost")
    assert_tensors_equal(results_2[1], expected_and_grad_2.expected2, "almost")
    assert_tensors_equal(results_2[2], expected_and_grad_2.expected3, "almost")
    assert_tensors_equal(
        GRADS[graph_2.nodes[0].inputs[0]], expected_and_grad_2.grad, "almost"
    )


fn main():
    print("Running dynamic ops (compare with torch) tests")
    try:
        test_CONCAT()
        test_SPLIT()
    except e:
        print("[ERROR] Error in dynamic ops (compare with torch)")
        print(e)
        return

    print("Finished dynamic ops (compare with torch) tests")
