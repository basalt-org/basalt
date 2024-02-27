from dainemo import dtype


struct Collection:
    var data: Pointer[Tensor[dtype]]
    var size: Int
    var capacity: Int

    fn __init__(inout self, capacity: Int):
        self.data = self.data.alloc(capacity)
        self.size = 0
        self.capacity = capacity

    fn __del__(owned self):
        self.data.free()

    fn replace(inout self, i: Int, owned value: Tensor[dtype]):
        __get_address_as_lvalue(self.data.offset(i).address) = value^

    fn append(inout self, owned value: Tensor[dtype]):
        if self.size == self.capacity:    
            self.resize(self.capacity * 2) # Growth strategy: double the capacity
        __get_address_as_uninit_lvalue(self.data.offset(self.size).address) = value^
        self.size += 1

    fn offset(inout self, i: Int) -> Pointer[Tensor[dtype]]:
        return self.data.offset(i)

    fn resize(inout self, new_capacity: Int):
        print("[COLLECTION] Resize called")
        if new_capacity >= self.size:
            let new_data: Pointer[Tensor[dtype]]
            new_data = new_data.alloc(new_capacity)
            for i in range(self.size):
                __get_address_as_uninit_lvalue(new_data.offset(i).address) = __get_address_as_lvalue(self.data.offset(i).address)
            self.data.free()
            self.data = new_data
            self.capacity = new_capacity
        else:
            print("Can't resize collection to capcity smaller then current size.")

    fn print(self, i: Int):
        print(__get_address_as_lvalue(self.data.offset(i).address))