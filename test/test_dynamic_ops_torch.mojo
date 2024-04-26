from random import rand
from python.python import Python

import basalt.nn as nn
from basalt import Graph, Symbol, OP
from basalt import Tensor, TensorShape
from basalt.autograd.attributes import Attribute, AttributeVector

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


fn torch_cat_op(
    op: OP, input_1: Tensor, input_2: Tensor, upper_grad: Tensor, dim: Int
) -> torch_output_cat:
    try:
        var py = Python.import_module("builtins")
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")

        var input_1 = torch.from_numpy(to_numpy(input_1)).requires_grad_(True)
        var input_2 = torch.from_numpy(to_numpy(input_2)).requires_grad_(True)

        var expected: PythonObject

        if op == OP.CONCAT:
            var tensors = py.list()
            tensors.append(input_1)
            tensors.append(input_2)
            expected = torch.cat(tensors, dim=dim)
        else:
            print("Error: op not supported, ", op)
            expected = input_1

        # uppergrad & backwards
        var upper_grad = torch.from_numpy(to_numpy(upper_grad))
        _ = expected.backward(upper_grad)

        return torch_output_cat(
            to_tensor(expected.detach().numpy()),
            to_tensor(input_1.grad.numpy()),
            to_tensor(input_2.grad.numpy()),
        )

    except e:
        print("Error importing torch: ", e)
        var d = Tensor[dtype](1)
        return torch_output_cat(d, d, d)


fn test_CONCAT() raises:
    alias t1_shape = TensorShape(20, 28)
    alias t2_shape = TensorShape(20, 28)
    var t1 = Tensor[dtype](t1_shape)
    var t2 = Tensor[dtype](t2_shape)
    rand(t1.data(), t1.num_elements())
    rand(t2.data(), t2.num_elements())

    # # default: dim = 0
    # alias ug_shape = TensorShape(40, 28) 
    # var ug = Tensor[dtype](ug_shape)
    # rand(ug.data(), ug.num_elements())

    # var expected_and_grad = torch_cat_op(OP.CONCAT, t1, t2, ug, dim=0)
    # test_binary_op[OP.CONCAT, t1_shape, t2_shape](
    #     t1, t2, expected_and_grad.expected
    # )
    # test_binary_op_backward[OP.CONCAT, t1_shape, t2_shape, ug_shape](
    #     t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    # )

    # # dim = 1
    # alias ug_shape_1 = TensorShape(20, 56)
    # ug = Tensor[dtype](ug_shape_1)
    # rand(ug.data(), ug.num_elements())

    # expected_and_grad = torch_cat_op(OP.CONCAT, t1, t2, ug, dim=1)
    # test_binary_op[OP.CONCAT, t1_shape, t2_shape, AttributeVector(Attribute("dim", 1))](
    #     t1, t2, expected_and_grad.expected
    # )
    # test_binary_op_backward[OP.CONCAT, t1_shape, t2_shape, ug_shape_1, AttributeVector(Attribute("dim", 1))](
    #     t1, t2, ug, expected_and_grad.grad_1, expected_and_grad.grad_2
    # )




fn main():
    print("Running dynamic ops (compare with torch) tests")
    try:
        test_CONCAT()
    except e:
        print("[ERROR] Error in dynamic ops (compare with torch)")
        print(e)
        return

    print("Finished dynamic ops (compare with torch) tests")