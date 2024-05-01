from python import Python
from pathlib import Path

from basalt.nn.model import Parameters
from basalt.nn.tensor import Tensor, TensorShape


fn load_onnx_data(
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
