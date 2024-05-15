from python.python import Python, PythonObject

import basalt.nn as nn
from basalt import dtype, Graph
from basalt import Tensor, TensorShape
from tests import assert_tensors_equal, to_numpy, to_tensor


fn test_upsample[
    shape: TensorShape,
    mode: StringLiteral,
    scale_factors: List[Scalar[dtype]],
    align_corners: Bool
](
    t1: Tensor[dtype],
    ug: Tensor[dtype],
    expected: Tensor[dtype],
    expected_grad: Tensor[dtype]
) raises:

    fn create_graph() -> Graph:
        var g = Graph()
        var t1 = g.input(shape, trainable=True)
        var t2 = nn.Upsample(g, t1, mode, scale_factors, align_corners)
        g.out(t2)
        return g ^

    alias graph = create_graph()
    var model = nn.Model[graph](inference_only=True)
    var res = model.inference(t1)[0]

    model.backward(ug)
    var res_grad = model.parameters.grads[graph.inputs[0]]

    assert_tensors_equal["almost"](res, expected)
    assert_tensors_equal["almost"](res_grad, expected_grad)


@value
struct torch_upsample_result:
    var expected: Tensor[dtype]
    var grad: Tensor[dtype]


fn test_upsample_torch[
    shape: TensorShape,
    mode: StringLiteral,
    scale_factors: List[Scalar[dtype]],
    align_corners: Bool
](data: PythonObject, ug: PythonObject) raises -> torch_upsample_result:

    var py = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    var torch = Python.import_module("torch")

    var py_scales = py.list()
    for i in range(len(scale_factors)):
        py_scales.append(scale_factors[i])

    # if mode == "nearest":
        # var ups = torch.nn.Upsample(scale_factor=py.tuple(py_scales), mode=mode)
    # else:
        # var ups = torch.nn.Upsample(scale_factor=py.tuple(py_scales), mode=mode, align_corners=align_corners)

    var ups = torch.nn.Upsample(scale_factor=py.tuple(py_scales), mode=mode)

    var tensor = torch.from_numpy(data).requires_grad_(True)
    var expected = ups(tensor)
    var upper_grad = torch.from_numpy(ug)
    _ = expected.backward(upper_grad)

    return torch_upsample_result(
        to_tensor(expected.detach().numpy()),
        to_tensor(tensor.grad.numpy()),
    )



fn test_UPSAMPLE_nearest() raises:
    var np = Python.import_module("numpy")

    alias shape = TensorShape(1, 1, 2, 2)
    alias mode: StringLiteral = "nearest"
    alias scales = List[Scalar[dtype]](2.0, 3.0)
    alias align_corners = False

    var data = np.array([
        1, 2,
        3, 4
    ], dtype=np.float32).reshape(1, 1, 2, 2)

    var ug = np.ones((1, 1, 4, 6))
    
    var torch_out = test_upsample_torch[shape, mode, scales, align_corners](data, ug)
    test_upsample[shape, mode, scales, align_corners](
        to_tensor(data),
        to_tensor(ug),
        torch_out.expected,
        torch_out.grad
    )

    _ = data


fn test_UPSAMPLE_linear() raises:
    var np = Python.import_module("numpy")

    alias shape = TensorShape(1, 1, 2, 2)
    alias mode: StringLiteral = "linear"
    alias scales = List[Scalar[dtype]](2.0, 2.0)

    var data = np.array([
        1, 2,
        3, 4
    ], dtype=np.float32).reshape(1, 1, 2, 2)

    # var expected = np.array([
    #     1.,   1.25, 1.75, 2.  ,
    #     1.5,  1.75, 2.25, 2.5 ,
    #     2.5,  2.75, 3.25, 3.5 ,
    #     3.,   3.25, 3.75, 4.  ,
    # ], dtype=np.float32).reshape(1, 1, 4, 4)


fn test_UPSAMPLE_cubic() raises:
    var np = Python.import_module("numpy")

    alias shape = TensorShape(1, 1, 4, 4)
    alias mode: StringLiteral = "cubic"
    alias scales = List[Scalar[dtype]](2.0, 2.0)

    var data = np.array([
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    ], dtype=np.float32).reshape(1, 1, 4, 4)

    # var expected = np.array([
    #      0.47265625,  0.76953125,  1.24609375,  1.875,       2.28125,   2.91015625,  3.38671875,  3.68359375,
    #      1.66015625,  1.95703125,  2.43359375,  3.0625,      3.46875,   4.09765625,  4.57421875,  4.87109375,
    #      3.56640625,  3.86328125,  4.33984375,  4.96875,     5.375,     6.00390625,  6.48046875,  6.77734375,
    #      6.08203125,  6.37890625,  6.85546875,  7.484375,    7.890625,  8.51953125,  8.99609375,  9.29296875,
    #      7.70703125,  8.00390625,  8.48046875,  9.109375,    9.515625, 10.14453125, 10.62109375, 10.91796875,
    #     10.22265625, 10.51953125, 10.99609375, 11.625,      12.03125,  12.66015625, 13.13671875, 13.43359375,
    #     12.12890625, 12.42578125, 12.90234375, 13.53125,    13.9375,   14.56640625, 15.04296875, 15.33984375,
    #     13.31640625, 13.61328125, 14.08984375, 14.71875,    15.125,    15.75390625, 16.23046875, 16.52734375
    # ], dtype=np.float32).reshape(1, 1, 8, 8)


fn main():

    try:
        test_UPSAMPLE_nearest()
        # test_UPSAMPLE_linear()
        # test_UPSAMPLE_cubic()
    except e:
        print("[Error] Error in Upsample")
        print(e)