from random import rand
from python.python import Python
from test_tensorutils import assert_tensors_equal

import basalt.nn as nn
from basalt import Graph, Symbol, OP, GRADS
from basalt import Tensor, TensorShape
from basalt.autograd.attributes import Attribute, AttributeVector

from test_dynamic_ops import create_graph_concat
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
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[0]], expected_and_grad.grad_1, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[1]], expected_and_grad.grad_2, "almost")
    assert_tensors_equal(GRADS[graph.nodes[0].inputs[2]], expected_and_grad.grad_3, "almost")

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
    assert_tensors_equal(GRADS[graph_2.nodes[0].inputs[0]], expected_and_grad_2.grad_1, "almost")
    assert_tensors_equal(GRADS[graph_2.nodes[0].inputs[1]], expected_and_grad_2.grad_2, "almost")
    assert_tensors_equal(GRADS[graph_2.nodes[0].inputs[2]], expected_and_grad_2.grad_3, "almost")


fn main():
    print("Running dynamic ops (compare with torch) tests")
    try:
        test_CONCAT()
    except e:
        print("[ERROR] Error in dynamic ops (compare with torch)")
        print(e)
        return

    print("Finished dynamic ops (compare with torch) tests")