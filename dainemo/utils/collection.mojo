'''
This currently serves as a replacement for a stdlib's List that can contain any types.
For every specific type a collection is defined that allows you to:
- Append elements of custom type structs into a collection of dynamic shape
- Replace elements in the collection
- Copy the collection
- Iterate over the collection
'''

# TODO: Refactor codebase when Traits are available.
# (Currently mojo v0.5.0)



from tensor import Tensor
from dainemo.autograd.node import Node
from dainemo.autograd.node import GraphNode


struct NodeCollection[dtype: DType = DType.float32]:
    var data: Pointer[Node[dtype]]
    var size: Int
    var capacity: Int
    var _current_index: Int

    fn __init__(inout self):
        self.size = 0
        self.capacity = 1
        self.data = self.data.alloc(self.capacity)
        self._current_index = 0

    fn __del__(owned self):
        self.data.free()

    @staticmethod
    fn store(i: Int, data: Pointer[Node[dtype]], value: Node[dtype]):
        __get_address_as_uninit_lvalue(data.offset(i).address) = value

    fn get(self, i: Int) -> Node[dtype]:
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
        self._current_index = other._current_index
        
        let new_data: Pointer[Node[dtype]]
        new_data = new_data.alloc(other.capacity)
        for i in range(other.size):
            self.store(i, new_data, __get_address_as_lvalue(other.data.offset(i).address))
        self.data = new_data

    fn copy(borrowed self) -> Self:
        let result: Self
        result.__copyinit__(self)
        return result

    fn __len__(borrowed self) -> Int:
        '''Iterations left.'''
        return self.size - self._current_index

    fn __iter__(inout self) -> Self:
        self._current_index = 0
        return self

    fn __next__(inout self) -> Node[dtype]:
        let result = self.get(self._current_index)
        self._current_index += 1
        return result

    fn __add__(self, other: Self) -> Self:
        var result = self.copy()
        for i in range(other.size):
            result.append(other.get(i))
        return result

    fn __iadd__(inout self, other: Self):
        for i in range(other.size):
            self.append(other.get(i))

    fn remove(inout self, idx: Int):
        for i in range(idx + 1, self.size):
            self.replace(i - 1, self.get(i))
        self.size -= 1

        # Free memory of unused space
        if self.size <= self.capacity // 4:
            self.resize(self.capacity // 2)

    # Extra (depends on Node struct)
    fn get_idx_by_uuid(inout self, uuid: String) -> Int:
        '''
        Returns the index of the GraphNode with the given uuid.
        When the uuid is not found in the collection, returns -1.
        '''
        for i in range(self.size):
            if __get_address_as_lvalue(self.data.offset(i).address).uuid == uuid:
                return i
        return -1

    fn set_grad_value(inout self, idx: Int, value: Tensor[dtype]):
        '''
        Sets the Node's gradient at the given index as <value> in the graph.
        '''
        __get_address_as_lvalue(self.data.offset(idx).address).grad = value

    fn get_grad_value(inout self, idx: Int) -> Tensor[dtype]:
        '''
        Get the Node's gradient at the given index in the graph.
        '''
        return __get_address_as_lvalue(self.data.offset(idx).address).grad



"""
Copy of NodeCollection, but with GraphNode instead of Node.
# TODO: Refactor to combine with variable type.
"""

struct GraphNodeCollection[dtype: DType = DType.float32]:
    var data: Pointer[GraphNode[dtype]]
    var size: Int
    var capacity: Int
    var _current_index: Int

    fn __init__(inout self):
        self.size = 0
        self.capacity = 1
        self.data = self.data.alloc(self.capacity)
        self._current_index = 0

    fn __del__(owned self):
        self.data.free()

    @staticmethod
    fn store(i: Int, data: Pointer[GraphNode[dtype]], value: GraphNode[dtype]):
        __get_address_as_uninit_lvalue(data.offset(i).address) = value

    fn get(self, i: Int) -> GraphNode[dtype]:
        return __get_address_as_lvalue(self.data.offset(i).address)

    fn replace(inout self, i: Int, value: GraphNode[dtype]):
        __get_address_as_lvalue(self.data.offset(i).address) = value

    fn resize(inout self, new_capacity: Int):
        if new_capacity >= self.size:
            let new_data: Pointer[GraphNode[dtype]]
            new_data = new_data.alloc(new_capacity)
            for i in range(self.size):
                self.store(i, new_data, self.get(i))
            self.data.free()
            self.data = new_data
            self.capacity = new_capacity
        else:
            print("Can't resize collection to capcity smaller then current size.")

    fn append(inout self, value: GraphNode[dtype]):
        if self.size == self.capacity:
            # Growth strategy: Double if limit is reached
            self.resize(self.capacity * 2)
        self.store(self.size, self.data, value)
        self.size += 1

    fn __copyinit__(inout self, other: GraphNodeCollection[dtype]):
        self.capacity = other.capacity
        self.size = other.size
        self._current_index = other._current_index
        
        let new_data: Pointer[GraphNode[dtype]]
        new_data = new_data.alloc(other.capacity)
        for i in range(other.size):
            self.store(i, new_data, __get_address_as_lvalue(other.data.offset(i).address))
        self.data = new_data

    fn copy(borrowed self) -> Self:
        let result: Self
        result.__copyinit__(self)
        return result

    fn __len__(borrowed self) -> Int:
        '''Iterations left.'''
        return self.size - self._current_index

    fn __iter__(inout self) -> Self:
        self._current_index = 0
        return self

    fn __next__(inout self) -> GraphNode[dtype]:
        let result = self.get(self._current_index)
        self._current_index += 1
        return result

    fn __add__(self, other: Self) -> Self:
        var result = self.copy()
        for i in range(other.size):
            result.append(other.get(i))
        return result

    fn __iadd__(inout self, other: Self):
        for i in range(other.size):
            self.append(other.get(i))

    fn remove(inout self, idx: Int):
        for i in range(idx + 1, self.size):
            self.replace(i - 1, self.get(i))
        self.size -= 1

        # Free memory of unused space
        if self.size <= self.capacity // 4:
            self.resize(self.capacity // 2)


    # Extra to avoid a copy & replacement in the GraphNodeCollection
    # (depends on GraphNode struct)
    fn set_visit_value(inout self, idx: Int, value: Bool):
        '''
        Marks the GraphNode at the given index as <value> in the graph.
        '''
        __get_address_as_lvalue(self.data.offset(idx).address).visited = value

    fn get_visit_value(inout self, idx: Int) -> Bool:
        '''
        Get the GraphNode's visit value at the given index in the graph.
        '''
        return __get_address_as_lvalue(self.data.offset(idx).address).visited

    fn set_grad_value(inout self, idx: Int, value: Tensor[dtype]):
        '''
        Sets the GraphNode's gradient at the given index as <value> in the graph.
        '''
        __get_address_as_lvalue(self.data.offset(idx).address).node.grad = value

    fn get_grad_value(inout self, idx: Int) -> Tensor[dtype]:
        '''
        Get the GraphNode's gradient at the given index in the graph.
        '''
        return __get_address_as_lvalue(self.data.offset(idx).address).node.grad
    
    fn get_idx_by_uuid(inout self, uuid: String) -> Int:
        '''
        Returns the index of the GraphNode with the given uuid.
        When the uuid is not found in the collection, returns -1.
        '''
        for i in range(self.size):
            if __get_address_as_lvalue(self.data.offset(i).address).node.uuid == uuid:
                return i
        return -1
