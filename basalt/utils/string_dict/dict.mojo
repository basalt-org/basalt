# This code is based on https://github.com/mzaks/compact-dict/tree/main

from math.bit import bit_length, ctpop
from memory import memset_zero, memcpy
from .keys_container import KeysContainer
from .ahasher import ahash


@always_inline
fn eq(a: StringRef, b: String) -> Bool:
    var l = len(a)
    if l != len(b):
        return False
    var p1 = a.data
    var p2 = b._as_ptr()
    var offset = 0
    alias step = 16
    while l - offset >= step:
        var unequal = p1.simd_load[step](offset) != p2.simd_load[step](offset)
        if unequal.reduce_or():
            return False
        offset += step
    while l - offset > 0:
        if p1.load(offset) != p2.load(offset):
            return False
        offset += 1
    return True


struct StringDict[
    V: CollectionElement, 
    hash: fn(String) -> UInt64 = ahash,
    KeyCountType: DType = DType.uint32,
    KeyOffsetType: DType = DType.uint32,
    destructive: Bool = True,
    caching_hashes: Bool = True,
](Sized):
    var keys: KeysContainer[KeyOffsetType]
    var key_hashes: DTypePointer[KeyCountType]
    var values: DynamicVector[V]
    var key_map: DTypePointer[KeyCountType]
    var deleted_mask: DTypePointer[DType.uint8]
    var count: Int
    var capacity: Int

    fn __init__(inout self, capacity: Int = 16):
        constrained[
            KeyCountType == DType.uint8 or 
            KeyCountType == DType.uint16 or 
            KeyCountType == DType.uint32 or 
            KeyCountType == DType.uint64,
            "KeyCountType needs to be an unsigned integer"
        ]()
        self.count = 0
        if capacity <= 8:
            self.capacity = 8
        else:
            var icapacity = Int64(capacity)
            self.capacity = capacity if ctpop(icapacity) == 1 else 1 << (bit_length(icapacity)).to_int()
        self.keys = KeysContainer[KeyOffsetType](capacity)
        @parameter
        if caching_hashes:
            self.key_hashes = DTypePointer[KeyCountType].alloc(self.capacity)
        else:
            self.key_hashes = DTypePointer[KeyCountType].alloc(0)
        self.values = DynamicVector[V](capacity=capacity)
        self.key_map = DTypePointer[KeyCountType].alloc(self.capacity)
        memset_zero(self.key_map, self.capacity)
        @parameter
        if destructive:
            self.deleted_mask = DTypePointer[DType.uint8].alloc(self.capacity >> 3)
            memset_zero(self.deleted_mask, self.capacity >> 3)
        else:
            self.deleted_mask = DTypePointer[DType.uint8].alloc(0)            

    fn __copyinit__(inout self, existing: Self):
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = existing.keys
        @parameter
        if caching_hashes:
            self.key_hashes = DTypePointer[KeyCountType].alloc(self.capacity)
            memcpy(self.key_hashes, existing.key_hashes, self.capacity)
        else:
            self.key_hashes = DTypePointer[KeyCountType].alloc(0)
        self.values = existing.values
        self.key_map = DTypePointer[KeyCountType].alloc(self.capacity)
        memcpy(self.key_map, existing.key_map, self.capacity)
        @parameter
        if destructive:
            self.deleted_mask = DTypePointer[DType.uint8].alloc(self.capacity >> 3)
            memcpy(self.deleted_mask, existing.deleted_mask, self.capacity >> 3)
        else:
            self.deleted_mask = DTypePointer[DType.uint8].alloc(0)

    fn __moveinit__(inout self, owned existing: Self):
        self.count = existing.count
        self.capacity = existing.capacity
        self.keys = existing.keys^
        self.key_hashes = existing.key_hashes
        self.values = existing.values^
        self.key_map = existing.key_map
        self.deleted_mask = existing.deleted_mask

    fn __del__(owned self):
        self.key_map.free()
        self.deleted_mask.free()
        self.key_hashes.free()

    fn __len__(self) -> Int:
        return self.count

    @always_inline
    fn __contains__(inout self, key: String) -> Bool:
        return self._find_key_index(key) != 0

    fn put(inout self, key: String, value: V):
        if self.count / self.capacity >= 0.87:
            self._rehash()
        
        var key_hash = hash(key).cast[KeyCountType]()
        var modulo_mask = self.capacity - 1
        var key_map_index = (key_hash & modulo_mask).to_int()
        while True:
            var key_index = self.key_map.load(key_map_index).to_int()
            if key_index == 0:
                self.keys.add(key)
                @parameter
                if caching_hashes:
                    self.key_hashes.store(key_map_index, key_hash)
                self.values.push_back(value)
                self.count += 1
                self.key_map.store(key_map_index, SIMD[KeyCountType, 1](self.keys.count))
                return
            @parameter
            if caching_hashes:
                var other_key_hash = self.key_hashes[key_map_index]
                if other_key_hash == key_hash:
                    var other_key = self.keys[key_index - 1]
                    if eq(other_key, key):
                        self.values[key_index - 1] = value # replace value
                        if destructive:
                            if self._is_deleted(key_index - 1):
                                self.count += 1
                                self._not_deleted(key_index - 1)
                        return
            else:
                var other_key = self.keys[key_index - 1]
                if eq(other_key, key):
                    self.values[key_index - 1] = value # replace value
                    if destructive:
                        if self._is_deleted(key_index - 1):
                            self.count += 1
                            self._not_deleted(key_index - 1)
                    return
            
            key_map_index = (key_map_index + 1) & modulo_mask

    @always_inline
    fn _is_deleted(self, index: Int) -> Bool:
        var offset = index >> 3
        var bit_index = index & 7
        return self.deleted_mask.offset(offset).load() & (1 << bit_index) != 0

    @always_inline
    fn _deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask.offset(offset)
        var mask = p.load()
        p.store(mask | (1 << bit_index))
    
    @always_inline
    fn _not_deleted(self, index: Int):
        var offset = index >> 3
        var bit_index = index & 7
        var p = self.deleted_mask.offset(offset)
        var mask = p.load()
        p.store(mask & ~(1 << bit_index))

    @always_inline
    fn _rehash(inout self):
        var old_key_map = self.key_map
        var old_capacity = self.capacity
        self.capacity <<= 1
        var mask_capacity = self.capacity >> 3
        self.key_map = DTypePointer[KeyCountType].alloc(self.capacity)
        memset_zero(self.key_map, self.capacity)
        
        var key_hashes = self.key_hashes
        @parameter
        if caching_hashes:
            key_hashes = DTypePointer[KeyCountType].alloc(self.capacity)
            
        @parameter
        if destructive:
            var deleted_mask = DTypePointer[DType.uint8].alloc(mask_capacity)
            memset_zero(deleted_mask, mask_capacity)
            memcpy(deleted_mask, self.deleted_mask, old_capacity >> 3)
            self.deleted_mask.free()
            self.deleted_mask = deleted_mask

        var modulo_mask = self.capacity - 1
        for i in range(old_capacity):
            if old_key_map[i] == 0:
                continue
            var key_hash = SIMD[KeyCountType, 1](0)
            @parameter
            if caching_hashes:
                key_hash = self.key_hashes[i]
            else:
                key_hash = hash(self.keys[(old_key_map[i] - 1).to_int()]).cast[KeyCountType]()

            var key_map_index = (key_hash & modulo_mask).to_int()

            var searching = True
            while searching:
                var key_index = self.key_map.load(key_map_index).to_int()

                if key_index == 0:
                    self.key_map.store(key_map_index, old_key_map[i])
                    searching = False
                else:
                    key_map_index = (key_map_index + 1) & modulo_mask
            @parameter
            if caching_hashes:
                key_hashes[key_map_index] = key_hash  
        
        @parameter
        if caching_hashes:
            self.key_hashes.free()
            self.key_hashes = key_hashes
        old_key_map.free()

    fn get(self, key: String, default: V) -> V:
        var key_index = self._find_key_index(key)
        if key_index == 0:
            return default

        @parameter
        if destructive: 
            if self._is_deleted(key_index - 1):
                return default
        return self.values[key_index - 1]

    fn delete(inout self, key: String):
        @parameter
        if not destructive:
            return

        var key_index = self._find_key_index(key)
        if key_index == 0:
                return
        if not self._is_deleted(key_index - 1):
            self.count -= 1
        self._deleted(key_index - 1)

    @always_inline
    fn _find_key_index(self, key: String) -> Int:
        var key_hash = hash(key).cast[KeyCountType]()
        var modulo_mask = self.capacity - 1

        var key_map_index = (key_hash & modulo_mask).to_int()
        while True:
            var key_index = self.key_map.load(key_map_index).to_int()
            if key_index == 0:
                return key_index
            
            @parameter
            if caching_hashes:
                var other_key_hash = self.key_hashes[key_map_index]
                if key_hash == other_key_hash:
                    var other_key = self.keys[key_index - 1]
                    if eq(other_key, key):
                        return key_index
            else:
                var other_key = self.keys[key_index - 1]
                if eq(other_key, key):
                    return key_index
            
            key_map_index = (key_map_index + 1) & modulo_mask

    fn debug(self):
        print("Dict count:", self.count, "and capacity:", self.capacity)
        print("KeyMap:")
        for i in range(self.capacity):
            print_no_newline(self.key_map.load(i))
            if i < self.capacity - 1:
                print_no_newline(", ")
            else:
                print("")
        print("Keys:")
        self.keys.print_keys()
        @parameter
        if caching_hashes:
            print("KeyHashes:")
            for i in range(self.capacity):
                if self.key_map.load(i) > 0:
                    print_no_newline(self.key_hashes.load(i))
                else:
                    print_no_newline(0)
                if i < self.capacity - 1:
                        print_no_newline(", ")
                else:
                    print("")