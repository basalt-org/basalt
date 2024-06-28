from python import Python

# maybe this functions should be from the Tensor struct (like tensor.to_numpy()) and tensor.__init__(np_array: PythonObject) to create a tensor from a numpy array and tensor.copy_np_data(np_array: PythonObject) to copy the numpy array to the tensor.


fn to_numpy(tensor: Tensor) -> PythonObject:
    try:
        var np = Python.import_module("numpy")

        np.set_printoptions(4)

        var rank = tensor.rank()
        var dims = PythonObject([])
        for i in range(rank):
            dims.append(tensor.dim(i))
        var pyarray: PythonObject = np.empty(dims, dtype=np.float32)

        var pointer = int(pyarray.__array_interface__["data"][0].to_float64())
        var pointer_d = DTypePointer[tensor.dtype](address=pointer)
        memcpy(pointer_d, tensor.data(), tensor.num_elements())

        _ = tensor

        return pyarray^
    except e:
        print("Error in to numpy", e)
        return PythonObject()


fn to_tensor(np_array: PythonObject) raises -> Tensor[dtype]:
    var shape = List[Int]()
    for i in range(np_array.ndim):
        shape.append(int(np_array.shape[i].to_float64()))
    if np_array.ndim == 0:
        # When the numpy array is a scalar, you need or the reshape to a size 1 ndarray or do this, if not the memcpy gets a memory error (Maybe because it is a register value?).
        var tensor = Tensor[dtype](TensorShape(1))
        tensor[0] = np_array.to_float64().cast[dtype]()
        return tensor^

    var tensor = Tensor[dtype](TensorShape(shape))

    var np_array_2: PythonObject
    try:
        var np = Python.import_module("numpy")
        # copy is also necessary for ops like slices to make them contiguous instead of references.
        np_array_2 = np.float32(np_array.copy())
    except e:
        np_array_2 = np_array.copy()
        print("Error in to_tensor", e)

    var pointer = int(np_array_2.__array_interface__["data"][0].to_float64())
    var pointer_d = DTypePointer[tensor.dtype](address=pointer)
    memcpy(tensor.data(), pointer_d, tensor.num_elements())

    _ = np_array_2
    _ = np_array

    return tensor^


fn copy_np_data(inout tensor: Tensor, np_array: PythonObject) raises:
    var np_array_2: PythonObject
    try:
        var np = Python.import_module("numpy")
        # copy is also necessary for ops like slices to make them contiguous instead of references.
        np_array_2 = np.float32(np_array.copy())
    except e:
        np_array_2 = np_array.copy()
        print("Error in to_tensor", e)

    var pointer = int(np_array_2.__array_interface__["data"][0].to_float64())
    var pointer_d = DTypePointer[tensor.dtype](address=pointer)
    memcpy(tensor.data(), pointer_d, tensor.num_elements())

    _ = np_array_2
    _ = np_array
    _ = tensor
