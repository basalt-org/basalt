from tensor import Tensor
from random import rand

import dainemo.nn as nn
from dainemo.utils.tensorutils import zero, fill, dot, tprint


fn main():
    alias dtype = DType.float32
    alias nelts: Int = simdwidthof[dtype]()

    # <------------LINEAR------------>
    let inputs: Tensor[dtype] = rand[dtype](2, 5)
    
    let f = nn.Linear[dtype](5, 3)

    tprint[dtype](inputs)
