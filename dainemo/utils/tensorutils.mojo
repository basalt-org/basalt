from tensor import Tensor, TensorShape
from utils.index import Index
from algorithm import vectorize, parallelize
from memory import memset_zero


@always_inline
fn zero[dtype: DType](inout t: Tensor[dtype]):
    memset_zero[dtype](t.data(), t.num_elements())


@always_inline
fn fill[dtype: DType, nelts: Int](inout t: Tensor[dtype], val: SIMD[dtype, 1]):
    @parameter
    fn fill_vec[nelts: Int](idx: Int):
        t.simd_store[nelts](idx, t.simd_load[nelts](idx).splat(val))
    vectorize[nelts, fill_vec](t.num_elements())


@always_inline
fn dot[dtype: DType, nelts: Int](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    let C = Tensor[dtype](A.dim(0), B.dim(1))
    memset_zero[dtype](C.data(), C.num_elements())  
    
    @parameter
    fn calc_row(m: Int):
        for k in range(A.dim(1)):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.simd_store[nelts](
                    m * C.dim(1) + n, 
                    C.simd_load[nelts](m * C.dim(1) + n) + A[m, k] * B.simd_load[nelts](k * B.dim(1) + n)
                )

            vectorize[nelts, dot](C.dim(1))

    parallelize[calc_row](C.dim(0), 20)

    return C


fn tprint[dtype: DType](t: Tensor[dtype], indent: Int = 0):
    let n: Int = t.num_elements()
    let shape = t.shape()
    var s: String

    if t.rank() == 0:
        s = String(t[0])
        print(s)
    elif t.rank() == 1:
        s = "[" + String(t[0])
        for i in range(1, shape[0]):
            s += "\t" + String(t[i])
        s += "]"
        print(s)
    #TODO: Implement recursive from here
    # else:
    #     print(repeat_tab(indent), "[")
    #     for i in range(shape[0]):
    #         ## TODO: select sub tensor of lower rank
    #         # tprint[dtype](sub_tensor, indent + 1)
            
    #     print(repeat_tab(indent), "]")
    
    elif t.rank() == 2:
        var srow: String
        
        s = "["
        for i in range(shape[0]):
            srow = "[" + String(t[i, 0])
            for j in range(1, shape[1]):
                srow += "\t" + String(t[i, j])
            srow += "]\n "
            s += srow
        s = s[:-2] + "]"
        print(s)

    elif t.rank() == 3:
        var smat: String
        var srow: String

        s = "[\n"
        for i in range(shape[0]):
            smat = "    ["
            for j in range(shape[1]):
                srow = "[" + String(t[i, j, 0])
                for k in range(1, shape[2]):
                    srow += "\t" + String(t[i, j, k])
                srow += "]\n     "
                smat += srow
            smat = smat[:-6] + "]"
            s += smat + "\n\n"
        s = s[:-1] + "]"
        print(s)
            
    print_no_newline("Tensor shape:", t.shape().__str__(), ", ")
    print_no_newline("Tensor rank:", t.rank(), ", ")
    print_no_newline("DType:", t.type().__str__(), "\n\n")