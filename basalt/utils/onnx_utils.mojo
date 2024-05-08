from python import Python
from pathlib import Path
from collections import Set

from basalt.nn.model import Parameters
from basalt.nn.tensor import Tensor, TensorShape
from basalt.autograd.attributes import Attribute, AttributeType
from basalt.autograd.ops import OP

# NOTE: Maybe we could create our own model representation and from there convert to onnx or others (well we already have it in reallity)
# NOTE: Torch doesn't import onnx, need onnx2torch and it doesn't support operators like reshape?

fn to_numpy(tensor: Tensor) raises -> PythonObject:
    var np = Python.import_module("numpy")

    np.set_printoptions(4)
    var rank = tensor.rank()
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


fn make_onnx_attribute(op: OP, attr: Attribute) raises -> PythonObject:
    var onnx = Python.import_module("onnx")
    var onnx_helper = Python.import_module("onnx.helper")

    var attr_name = str(attr.name)
    var attr_value: PythonObject

    if attr.type == AttributeType.FLOAT:
        attr_value = attr.to_scalar[DType.float64]()
    elif attr.type == AttributeType.INT:
        attr_value = attr.to_int()
    elif attr.type == AttributeType.STRING:
        attr_value = attr.to_string()
    elif attr.type == AttributeType.INTS:
        var temp = attr.to_shape()
        var shape = PythonObject([])
        for i in range(temp.rank()):
            shape.append(temp[i])
        attr_value = shape
    else:
        raise Error("Unsupported attribute type")

    if op == OP.CONV2D or op == OP.MAXPOOL2D:
        if attr_name == "dilation":
            attr_name = "dilations"
        elif attr_name == "kernel_size":
            attr_name = "kernel_shape"
        elif attr_name == "stride":
            attr_name = "strides"
        elif attr_name == "padding":
            attr_name = "pads"
        else:
            raise Error("Unsupported attribute name for operator " + str(op))

    if (op == OP.CONV2D and attr_name) == "pads" or (
        op == OP.MAXPOOL2D and attr_name
    ) == "pads":
        # Special case for pads in conv and maxpool, onnx wants pads to be [x1_begin, x2_begin…x1_end, x2_end,…],
        attr_value.append(attr_value[0])
        attr_value.append(attr_value[1])

    return onnx_helper.make_attribute(attr_name, attr_value)


fn make_onnx_operator_type(op_type: OP) raises -> String:
    if op_type == OP.ADD:
        return "Add"
    elif op_type == OP.SUB:
        return "Sub"
    elif op_type == OP.MUL:
        return "Mul"
    elif op_type == OP.DOT:
        return "MatMul"
    elif op_type == OP.DIV:
        return "Div"
    elif op_type == OP.EXP:
        return "Exp"
    elif op_type == OP.LOG:
        return "Log"
    elif op_type == OP.SUM:
        # Special case, axis isn't an attribute, instead it is an input, because it can be dynamic
        raise Error(str(op_type) + " is not supported right now for conversion to onnx")
        # return "ReduceSum"
    elif op_type == OP.MEAN:
        raise Error(str(op_type) + " is not supported right now for conversion to onnx")
        # return "ReduceMean"
    elif op_type == OP.MAX:
        raise Error(str(op_type) + " is not supported right now for conversion to onnx")
        # return "ReduceMax"
    elif op_type == OP.CONV2D:
        return "Conv"
    elif op_type == OP.MAXPOOL2D:
        return "MaxPool"
    elif op_type == OP.RELU:
        return "Relu"
    elif op_type == OP.TANH:
        return "Tanh"
    elif op_type == OP.SIGMOID:
        return "Sigmoid"
    elif op_type == OP.RESHAPE:
        return "Reshape"
    elif op_type == OP.TRANSPOSE:
        return "Transpose"
    elif op_type == OP.FLATTEN:
        return "Flatten"
    elif op_type == OP.SQUEEZE:
        return "Squeeze"
    elif op_type == OP.UNSQUEEZE:
        return "Unsqueeze"
    elif op_type == OP.CONCAT:
        return "Concat"
    elif op_type == OP.SPLIT:
        return "Split"
    elif op_type == OP.CLIP:
        return "Clip"
    elif op_type == OP.FMA:
        raise Error(str(op_type) + " operator is not supported in onnx")
    else:
        raise Error("Unsupported operator type " + str(op_type))


# --- Loader and exporter ---
fn load_onnx_model(
    model_path: Path, inout model_parameters: Parameters, g: Graph
) raises:
    # Simple onnx data loader where we load the data in order (so we need to have the correct order of the weights and biases in the model. We don't use the names for the loading)
    var onnx = Python.import_module("onnx")
    var onnx_model = onnx.load(str(model_path))

    for i in range(len(onnx_model.graph.initializer)):
        var tensor = onnx_model.graph.initializer[i]

        if (
            tensor.data_type == onnx.TensorProto.FLOAT
            or tensor.data_type == onnx.TensorProto.INT32
            or tensor.data_type == onnx.TensorProto.INT64
        ):
            var data_np = onnx.numpy_helper.to_array(tensor)

            # Get the shape of data onnx
            var temp = List[Int]()
            for j in range(len(data_np.shape)):
                temp.append(int(data_np.shape[j].to_float64()))
            var data_shape = TensorShape(temp)

            # Compare the shape of the data with the shape of the model tensor
            var model_tensor_shape = g.params.symbols[i].shape

            if data_shape != model_tensor_shape:
                # check if the shape is transposed (reversed), we do this comparison because torch can save sove weights transposed (like gemm operator)

                var raise_error_flag = True
                if data_shape.rank() == model_tensor_shape.rank():
                    var count = 0
                    for j in range(model_tensor_shape.rank()):
                        if (
                            data_shape[data_shape.rank() - j - 1]
                            != model_tensor_shape[j]
                        ):
                            break
                        count += 1

                    if count == model_tensor_shape.rank():
                        raise_error_flag = False
                        data_np = data_np.transpose()

                if raise_error_flag:
                    raise Error(
                        "Shape mismatch for tensor "
                        + str(i)
                        + ". Expected shape: "
                        + model_tensor_shape
                        + ", got shape: "
                        + data_shape
                    )

            var data = data_np.flatten()

            # It would be better to use memcpy here
            for j in range(len(data)):
                model_parameters.tensors[g.params.symbols[i]][j] = data[j].to_float64()
        else:
            raise Error("Unsupported data type")


fn export_onnx_model(model_path: Path, model_parameters: Parameters, g: Graph) raises:
    # Create onnx model with data and nodes
    var onnx = Python.import_module("onnx")
    var onnx_helper = Python.import_module("onnx.helper")

    var graph = onnx_helper.make_graph(
        nodes=[],
        name="basalt_model",
        inputs=[],
        outputs=[],
        initializer=[],
        value_info=[],
    )

    var visited = Set[String]()

    # Create onnx initializers
    for i in range(len(g.params.symbols)):
        var tensor = g.params.symbols[i]
        var tensor_data = model_parameters.tensors[tensor]
        var tensor_np = to_numpy(tensor_data)

        # Create onnx tensor
        var onnx_tensor_data = onnx_helper.make_tensor(
            name=str(tensor.name),
            data_type=onnx.TensorProto.FLOAT,
            dims=tensor_np.shape,
            vals=tensor_np,
        )

        graph.initializer.append(onnx_tensor_data)

    # Create onnx nodes
    for i in range(len(g.nodes)):
        var node = g.nodes[i]

        var op_type = make_onnx_operator_type(node.operator)
        var inputs = PythonObject([])
        var outputs = PythonObject([])
        var name = str(node.operator) + "_node" + i

        for j in range(len(node.inputs)):
            inputs.append(str(node.inputs[j].name))
        for j in range(len(node.outputs)):
            outputs.append(str(node.outputs[j].name))

            # Process intermediate
            if str(node.outputs[j].name) not in visited:
                visited.add(str(node.outputs[j].name))
                var intermediate_tensor = node.outputs[j]
                var intermediate_shape = intermediate_tensor.shape

                var name = str(intermediate_tensor.name)
                var dtype = onnx.TensorProto.FLOAT  # TODO
                var shape = PythonObject([])
                for j in range(intermediate_shape.rank()):
                    shape.append(intermediate_shape[j])

                # Create onnx tensor information
                var onnx_output = onnx_helper.make_tensor_value_info(name, dtype, shape)
                graph.value_info.append(onnx_output)

        # Create onnx node
        var onnx_node = onnx_helper.make_node(
            op_type,
            inputs,
            outputs,
            name,
        )

        # Process attributes
        for j in range(len(node.attributes)):
            var attr = node.attributes[j]
            var attr_value = make_onnx_attribute(node.operator, attr)

            # Special case for reshape, shape in reshape is not an attribute, instead it is an input because they can be dynamic
            if not node.operator == OP.RESHAPE:
                onnx_node.attribute.append(attr_value)

        # Special case for reshape, shape in reshape is not an attribute, instead it is an input because they can be dynamic (it can be the result of another operator, don't know why)
        if node.operator == OP.RESHAPE:
            var shape = node.attributes[0].to_shape()
            var list_shape = PythonObject([])
            for j in range(shape.rank()):
                list_shape.append(shape[j])

            graph.initializer.append(
                onnx_helper.make_tensor(
                    name=name + "_shape",
                    data_type=onnx.TensorProto.INT64,
                    dims=(shape.rank(),),
                    vals=list_shape,
                )
            )

            onnx_node.input.append(name + "_shape")

        graph.node.append(onnx_node)

    # Create onnx inputs
    for i in range(len(g.inputs)):
        var input_tensor = g.inputs[i]
        var input_shape = input_tensor.shape

        var name = str(input_tensor.name)
        var dtype = onnx.TensorProto.FLOAT  # TODO
        var shape = PythonObject([])
        for j in range(input_shape.rank()):
            shape.append(input_shape[j])

        # Create onnx tensor information
        var onnx_input = onnx_helper.make_tensor_value_info(name, dtype, shape)
        graph.input.append(onnx_input)

    # Create onnx outputs
    for i in range(len(g.outputs)):
        var output_tensor = g.outputs[i]
        var output_shape = output_tensor.shape

        var name = str(output_tensor.name)
        var dtype = onnx.TensorProto.FLOAT  # TODO
        var shape = PythonObject([])
        for j in range(output_shape.rank()):
            shape.append(output_shape[j])

        # Create onnx tensor information
        var onnx_output = onnx_helper.make_tensor_value_info(name, dtype, shape)
        graph.output.append(onnx_output)

    # Create onnx model
    var onnx_model = onnx_helper.make_model(graph, producer_name="basalt")

    # Save onnx model
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(model_path))
