# This code is based on https://github.com/mzaks/compact-dict/tree/main

from collections.vector import InlinedFixedVector


struct KeysContainer[KeyEndType: DType = DType.uint32](Sized):
    var keys: DTypePointer[DType.int8]
    var allocated_bytes: Int
    var keys_end: DTypePointer[KeyEndType]
    var count: Int
    var capacity: Int

    fn __init__(inout self, capacity: Int):
        constrained[
            KeyEndType == DType.uint8
            or KeyEndType == DType.uint16
            or KeyEndType == DType.uint32
            or KeyEndType == DType.uint64,
            "KeyEndType needs to be an unsigned integer",
        ]()
        self.allocated_bytes = capacity << 3
        self.keys = DTypePointer[DType.int8].alloc(self.allocated_bytes)
        self.keys_end = DTypePointer[KeyEndType].alloc(capacity)
        self.count = 0
        self.capacity = capacity

    fn __copyinit__(inout self, existing: Self):
        self.allocated_bytes = existing.allocated_bytes
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = DTypePointer[DType.int8].alloc(self.allocated_bytes)
        memcpy(self.keys, existing.keys, self.allocated_bytes)
        self.keys_end = DTypePointer[KeyEndType].alloc(self.allocated_bytes)
        memcpy(self.keys_end, existing.keys_end, self.capacity)

    fn __moveinit__(inout self, owned existing: Self):
        self.allocated_bytes = existing.allocated_bytes
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = existing.keys
        self.keys_end = existing.keys_end

    fn __del__(owned self):
        self.keys.free()
        self.keys_end.free()

    @always_inline
    fn add(inout self, key: String):
        var prev_end = 0 if self.count == 0 else self.keys_end[self.count - 1]
        var key_length = len(key)
        var new_end = prev_end + key_length

        var needs_realocation = False
        while new_end > self.allocated_bytes:
            self.allocated_bytes += self.allocated_bytes >> 1
            needs_realocation = True

        if needs_realocation:
            var keys = DTypePointer[DType.int8].alloc(self.allocated_bytes)
            memcpy(keys, self.keys, prev_end.to_int())
            self.keys.free()
            self.keys = keys

        memcpy(self.keys.offset(prev_end), key._as_ptr(), key_length)
        var count = self.count + 1
        if count >= self.capacity:
            var new_capacity = self.capacity + (self.capacity >> 1)
            var keys_end = DTypePointer[KeyEndType].alloc(self.allocated_bytes)
            memcpy(keys_end, self.keys_end, self.capacity)
            self.keys_end.free()
            self.keys_end = keys_end
            self.capacity = new_capacity

        self.keys_end.store(self.count, new_end)
        self.count = count

    @always_inline
    fn get(self, index: Int) -> StringRef:
        if index < 0 or index >= self.count:
            return ""
        var start = 0 if index == 0 else self.keys_end[index - 1].to_int()
        var length = self.keys_end[index].to_int() - start
        return StringRef(self.keys.offset(start), length)

    @always_inline
    fn __getitem__(self, index: Int) -> StringRef:
        return self.get(index)

    @always_inline
    fn __len__(self) -> Int:
        return self.count

    fn keys_vec(self) -> InlinedFixedVector[StringRef]:
        var keys = InlinedFixedVector[StringRef](self.count)
        for i in range(self.count):
            keys.append(self[i])
        return keys

    fn print_keys(self):
        print_no_newline("(")
        print_no_newline(self.count)
        print_no_newline(")[")
        for i in range(self.count):
            print_no_newline(self[i])
            if i < self.count - 1:
                print_no_newline(", ")
        print("]")
