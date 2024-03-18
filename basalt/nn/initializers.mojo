from math import sqrt
from tensor import TensorShape

from basalt import dtype
from basalt.utils.rand_utils import rand_normal, rand_uniform


fn initialize_tensor(shape: TensorShape, type: String, data: DynamicVector[SIMD[dtype, 1]]) -> Tensor[dtype]:
    if type == "random_uniform":
        var low = data[0]
        var high = data[1]
        var t = Tensor[dtype](shape)
        rand_uniform(t, low = low, high = high)
        return t
    elif type == "random_normal":
        var mean = data[0].cast[DType.float64]()
        var std = data[1].cast[DType.float64]()
        var t = Tensor[dtype](shape)
        rand_normal(t, mean = mean, std = std)
        return t
    elif type == "kaiming_uniform":
        # mode, nonlinearity
        var mode_id = data[0]
        var mode = "fan_in" if mode_id == 0 else "fan_out"
        return kaiming_uniform(shape, mode = mode)
    elif type == "kaiming_normal":
        # mode, nonlinearity
        var mode_id = data[0]
        var mode = "fan_in" if mode_id == 0 else "fan_out"
        return kaiming_normal(shape, mode = mode)
    else:
        print("[ERROR] Unsupported initialization type: " + type)
        return Tensor[dtype](shape)


fn calculate_fan(shape: TensorShape, mode: String) -> SIMD[dtype, 1]:
    """
    Calculate the fan-in and fan-out of any tensor.
    """
    # NOTE: shape.rank() should be > 2
    # mode: "fan_in" or "fan_out"
    if shape.rank() < 2:
        print("[ERROR] Fan in and fan out can not be calculated for tensor with less than 2 dimensions")

    var num_input_fmaps = shape[1]
    var num_output_fmaps = shape[0]
    var receptive_field_size = 1
    if shape.rank() > 2:
        for i in range(2, shape.rank()):
            receptive_field_size *= shape[i]
        
    var fan_in = num_input_fmaps * receptive_field_size
    var fan_out = num_output_fmaps * receptive_field_size

    if mode == "fan_in":
        return fan_in
    else:
        return fan_out


# TODO: https://pytorch.org/docs/stable/_modules/torch/nn/init.html
fn kaiming_uniform(shape: TensorShape, mode: String = "fan_in", nonlinearity: String = "leaky_relu") -> Tensor[dtype]:
    var fan = calculate_fan(shape, mode)
   
    # TODO: add support for other gains: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py#L68
    # Gain for linear and conv layers is 1
    var gain = 1 
    var std = gain / sqrt(fan)

    # var bound = sqrt(3) * std.cast[dtype]()
    var bound = std.cast[dtype]()

    # print("Shape", shape, "Fan", fan, "Bound", bound)

    var t = Tensor[dtype](shape)
    rand_uniform(t, low = -bound, high = bound)
    return t^


# TODO: https://pytorch.org/docs/stable/_modules/torch/nn/init.html
fn kaiming_normal(shape: TensorShape, mode: String = "fan_in", nonlinearity: String = "leaky_relu") -> Tensor[dtype]:
    var fan = calculate_fan(shape, mode)

    # TODO: add support for other gains: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py#L68
    # Gain for linear and conv layers is 1
    var gain = 1
    var std = gain / sqrt(fan)
    
    var t = Tensor[dtype](shape)
    rand_normal(t, mean = 0, std = std.cast[DType.float64]())
    return t^