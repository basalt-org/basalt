from tensor import Tensor

from dainemo.utils.tensorutils import zero, fill, dot, tprint


fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()

    var A: Tensor[dtype] = Tensor[dtype](2, 3)
    var B: Tensor[dtype] = Tensor[dtype](3, 2)

    
    # <------------ZERO------------>
    zero[dtype](A)
    tprint[dtype](A)
    
    # <------------FILL------------>
    fill[dtype, nelts](A, 1.0)
    tprint[dtype](A)

    fill[dtype, nelts](B, 1.0)
    tprint[dtype](B)
    
    # <------------DOT------------>
    let C = dot[dtype, nelts](A, B)
    tprint[dtype](C)

