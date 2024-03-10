from math import sqrt
from tensor import TensorShape

from dainemo.utils.rand_utils import rand_normal, rand_uniform


fn initialize_tensor(shape: TensorShape, type: String) -> Tensor[dtype]:
    if type == "kaiming_uniform":
        return kaiming_uniform(shape, mode = "fan_in")
    elif type == "kaiming_normal":
        return kaiming_normal(shape, mode = "fan_in")
    else:
        print("[ERROR] Unsupported initialization type: " + type)
        return Tensor[dtype](shape)


fn calculate_fan(shape: TensorShape, mode: String) -> Int:
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


fn kaiming_uniform(shape: TensorShape, mode: String = "fan_in") -> Tensor[dtype]:
    var fan_in = calculate_fan(shape, mode)
   
    var gain = 1 # TODO: add support for other gains: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py#L68
    var std = 1 / sqrt(fan_in)
    
    var bound = sqrt(3) * std.cast[dtype]()
    var t = Tensor[dtype](shape)
    rand_uniform(t, low = -bound, high = bound)
    return t^


fn kaiming_normal(shape: TensorShape, mode: String = "fan_in") -> Tensor[dtype]:
    var fan_in = calculate_fan(shape, mode)
    
    var gain = 1 # TODO: add support for other gains: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py#L68
    var std = gain / sqrt(fan_in)
    
    var t = Tensor[dtype](shape)
    rand_normal(t, mean = 0, std = std.cast[DType.float64]())
    return t^