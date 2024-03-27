from python.python import Python
from basalt import Tensor, TensorShape


alias dtype = DType.float32


def to_numpy(tensor: Tensor) -> PythonObject:
    var np = Python.import_module("numpy")
    np.set_printoptions(4)

    rank = tensor.rank()
    var pyarray: PythonObject = np.array([0])
    if rank == 1:
        pyarray = np.empty((tensor.dim(0)))
    elif rank == 2:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1)))
    elif rank == 3:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2)))
    elif rank == 4:
        pyarray = np.empty((tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3)))
    else:
        print("Error: rank not supported: ", rank)

    for i in range(tensor.num_elements()):
        pyarray.itemset((i), tensor[i])

    return pyarray


fn to_tensor(np_array: PythonObject) raises -> Tensor[dtype]:
    var shape = DynamicVector[Int]()
    for i in range(np_array.ndim):
        shape.push_back(np_array.shape[i].to_float64().to_int())

    var tensor = Tensor[dtype](TensorShape(shape))

    # Calling ravel a lot of times is slow
    var np_array_temp = np_array.ravel()

    for i in range(tensor.num_elements()):
        tensor[i] = np_array_temp[i].to_float64().cast[dtype]()

    return tensor
