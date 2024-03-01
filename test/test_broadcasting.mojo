from python.python import Python
from tensor import TensorShape
from testing import assert_true

from dainemo.utils.tensorutils import broadcast_shapes


fn to_tensor_shape(owned shape: PythonObject) raises -> TensorShape:
    var tensor_shape = DynamicVector[Int]()
    for dim in shape:
        tensor_shape.push_back(dim.to_float64().to_int())
    return TensorShape(tensor_shape)


fn np_broadcast_shapes(s1: TensorShape, s2: TensorShape) raises -> TensorShape:
    var np = Python.import_module("numpy")
    # Convert to python list
    var s1_py: PythonObject = []
    var s2_py: PythonObject = []
    for i in range(s1.rank()):
        s1_py += [s1[i]]
    for i in range(s2.rank()):
        s2_py += [s2[i]]

    # Numpy broadcast_shapes
    var py_shape = np.broadcast_shapes(s1_py, s2_py)

    return to_tensor_shape(py_shape)


fn test_broadcast_shapes() raises:
    var s1 = TensorShape(3, 5, 2)
    var s2 = TensorShape(3, 5, 2)
    var s3 = broadcast_shapes(s1, s2)
    assert_true(s3 == np_broadcast_shapes(s1, s2))

    s1 = TensorShape(3, 5, 2)
    s2 = TensorShape(1, 2)
    s3 = broadcast_shapes(s1, s2)
    assert_true(s3 == np_broadcast_shapes(s1, s2))

    s1 = TensorShape(5, 1)
    s2 = TensorShape(3, 5, 1)
    s3 = broadcast_shapes(s1, s2)
    assert_true(s3 == np_broadcast_shapes(s1, s2))

    s1 = TensorShape(3, 1, 2)
    s2 = TensorShape(3, 5, 2)
    s3 = broadcast_shapes(s1, s2)
    assert_true(s3 == np_broadcast_shapes(s1, s2))

    s1 = TensorShape(1, 1, 1)
    s2 = TensorShape(3, 5, 2)
    s3 = broadcast_shapes(s1, s2)
    assert_true(s3 == np_broadcast_shapes(s1, s2))

    s1 = TensorShape(2)
    s2 = TensorShape(3, 5, 2)
    s3 = broadcast_shapes(s1, s2)
    assert_true(s3 == np_broadcast_shapes(s1, s2))

    s1 = TensorShape()
    s2 = TensorShape(3, 5, 2)
    s3 = broadcast_shapes(s1, s2)
    assert_true(s3 == np_broadcast_shapes(s1, s2))

    # # Both errors expected
    # print("EXPECTED RAISE!")
    # try: 
    #     s1 = TensorShape(3, 2, 2)
    #     s2 = TensorShape(3, 5, 2)
    #     s3 = broadcast_shapes(s1, s2)
    #     _ = np_broadcast_shapes(s1, s2)
    # except e:
    #     print("Numpy:", e)

    # print("EXPECTED RAISE!")
    # try: 
    #     s1 = TensorShape(3)
    #     s2 = TensorShape(2)
    #     s3 = broadcast_shapes(s1, s2)
    #     _ = np_broadcast_shapes(s1, s2)
    # except e:
    #     print("Numpy:", e)


fn test_broadcast_shapes_multiple() raises:
    var np = Python.import_module("numpy")

    var s1 = TensorShape(1, 2)
    var s2 = TensorShape(3, 1)
    var s3 = TensorShape(3, 2)
    var res = broadcast_shapes(s1, s2, s3)
    var res_np = to_tensor_shape(np.broadcast_shapes((1, 2), (3, 1), (3, 2)))
    assert_true(res == res_np)

    s1 = TensorShape(6, 7)
    s2 = TensorShape(5, 6, 1)
    s3 = TensorShape(7)
    var s4 = TensorShape(5, 1, 7)
    res = broadcast_shapes(s1, s2, s3, s4)
    res_np = to_tensor_shape(np.broadcast_shapes((6, 7), (5, 6, 1), (7), (5, 1, 7)))
    assert_true(res == res_np)






fn main():

    try:
        test_broadcast_shapes()
        test_broadcast_shapes_multiple()
    except e:
        print("[Error] In test broadcasting.")
        print(e)