from collections import Dict


fn modify_two_dict_values(inout d1: String, d2: String):
    d1 = "8"
    d2 = "9"

fn main() raises:
    var d = Dict[String, String]()
    d["one"] = "1"
    d["two"] = "2"
    d["three"] = "3"

    print(d["one"])
    print(d["two"])
    modify_two_dict_values(d["one"], d["two"])

    print(d["one"])
    print(d["two"])

    var ma = List[Int](1, 2)

    var h = Pointer.address_of(ma)