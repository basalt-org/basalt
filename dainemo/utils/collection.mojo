from dainemo.autograd.node import Node


struct NodeCollection[dtype: DType = DType.float32]:
    var data: Pointer[Node[dtype]]
    var size: Int
    var capacity: Int

    fn __init__(inout self):
        self.size = 0
        self.capacity = 1
        self.data = self.data.alloc(self.capacity)

    fn __moveinit__(inout self, owned other: Self):
        self.data = other.data
        self.capacity = other.capacity
        self.size = other.size

    fn __del__(owned self):
        self.data.free()

    @staticmethod
    fn store(i: Int, data: Pointer[Node[dtype]], value: Node[dtype]):
        __get_address_as_uninit_lvalue(data.offset(i).address) = value

    fn get(inout self, i: Int) -> Node[dtype]:
        return __get_address_as_lvalue(self.data.offset(i).address)

    fn replace(inout self, i: Int, value: Node[dtype]):
        __get_address_as_lvalue(self.data.offset(i).address) = value

    fn resize(inout self, new_capacity: Int):
        if new_capacity >= self.size:
            let new_data: Pointer[Node[dtype]]
            new_data = new_data.alloc(new_capacity)
            for i in range(self.size):
                self.store(i, new_data, self.get(i))
            self.data.free()
            self.data = new_data
            self.capacity = new_capacity
        else:
            print("Can't resize collection to capcity smaller then current size.")

    fn append(inout self, value: Node[dtype]):
        if self.size == self.capacity:
            # Growth strategy: Double if limit is reached
            self.resize(self.capacity * 2)
        self.store(self.size, self.data, value)
        self.size += 1

    fn __copyinit__(inout self, other: NodeCollection[dtype]):
        self.capacity = other.capacity
        self.size = other.size
        
        let new_data: Pointer[Node[dtype]]
        new_data = new_data.alloc(other.capacity)
        for i in range(other.size):
            self.store(i, new_data, __get_address_as_lvalue(other.data.offset(i).address))
        self.data = new_data

    fn copy(borrowed self) -> Self:
        let result: Self
        result.__copyinit__(self)
        return result
