from tensor import Tensor, TensorShape


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