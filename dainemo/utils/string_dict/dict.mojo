# This code is based on https://github.com/mzaks/compact-dict

from math.bit import bit_length, ctpop
from memory import memset_zero, memcpy
from .string_eq import eq
from .keys_container import KeysContainer
from .ahasher import ahash

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
            self.capacity = capacity if ctpop(icapacity) == 1 else
                            1 << (bit_length(icapacity)).to_int()
        self.keys = KeysContainer[KeyOffsetType](capacity)
        @parameter
        if caching_hashes:
            self.key_hashes = DTypePointer[KeyCountType].alloc(self.capacity)
        else:
            self.key_hashes = DTypePointer[KeyCountType].alloc(0)
        self.values = DynamicVector[V](self.capacity)
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

    fn put(inout self, key: String, value: V):
        if self.count / self.capacity >= 0.87:
            self._rehash()
        
        let key_hash = hash(key).cast[KeyCountType]()
        let modulo_mask = self.capacity - 1
        var key_map_index = (key_hash & modulo_mask).to_int()
        while True:
            let key_index = self.key_map.load(key_map_index).to_int()
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
                let other_key_hash = self.key_hashes[key_map_index]
                if other_key_hash == key_hash:
                    let other_key = self.keys[key_index - 1]
                    if eq(other_key, key):
                        self.values[key_index - 1] = value # replace value
                        if destructive:
                            if self._is_deleted(key_index - 1):
                                self.count += 1
                                self._not_deleted(key_index - 1)
                        return
            else:
                let other_key = self.keys[key_index - 1]
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
        let offset = index >> 3
        let bit_index = index & 7
        return self.deleted_mask.offset(offset).load() & (1 << bit_index) != 0

    @always_inline
    fn _deleted(self, index: Int):
        let offset = index >> 3
        let bit_index = index & 7
        let p = self.deleted_mask.offset(offset)
        let mask = p.load()
        p.store(mask | (1 << bit_index))
    
    @always_inline
    fn _not_deleted(self, index: Int):
        let offset = index >> 3
        let bit_index = index & 7
        let p = self.deleted_mask.offset(offset)
        let mask = p.load()
        p.store(mask & ~(1 << bit_index))

    @always_inline
    fn _rehash(inout self):
        let old_key_map = self.key_map
        let old_capacity = self.capacity
        self.capacity <<= 1
        let mask_capacity = self.capacity >> 3
        self.key_map = DTypePointer[KeyCountType].alloc(self.capacity)
        memset_zero(self.key_map, self.capacity)
        
        var key_hashes = self.key_hashes
        @parameter
        if caching_hashes:
            key_hashes = DTypePointer[KeyCountType].alloc(self.capacity)
            
        @parameter
        if destructive:
            let deleted_mask = DTypePointer[DType.uint8].alloc(mask_capacity)
            memset_zero(deleted_mask, mask_capacity)
            memcpy(deleted_mask, self.deleted_mask, old_capacity >> 3)
            self.deleted_mask.free()
            self.deleted_mask = deleted_mask

        let modulo_mask = self.capacity - 1
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
                let key_index = self.key_map.load(key_map_index).to_int()

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
        let key_hash = hash(key).cast[KeyCountType]()
        let modulo_mask = self.capacity - 1

        var key_map_index = (key_hash & modulo_mask).to_int()
        while True:
            let key_index = self.key_map.load(key_map_index).to_int()
            if key_index == 0:
                return default
            
            @parameter
            if caching_hashes:
                let other_key_hash = self.key_hashes[key_map_index]
                if key_hash == other_key_hash:
                    let other_key = self.keys[key_index - 1]
                    if eq(other_key, key):
                        if destructive: 
                            if self._is_deleted(key_index - 1):
                                return default
                        return self.values[key_index - 1]
            else:
                let other_key = self.keys[key_index - 1]
                if eq(other_key, key):
                    if destructive: 
                        if self._is_deleted(key_index - 1):
                            return default
                    return self.values[key_index - 1]
            
            key_map_index = (key_map_index + 1) & modulo_mask

    fn __contains__(self, key: String) -> Bool:
        let key_hash = hash(key).cast[KeyCountType]()
        let modulo_mask = self.capacity - 1

        var key_map_index = (key_hash & modulo_mask).to_int()
        while True:
            let key_index = self.key_map.load(key_map_index).to_int()
            if key_index == 0:
                return False
            
            @parameter
            if caching_hashes:
                let other_key_hash = self.key_hashes[key_map_index]
                if key_hash == other_key_hash:
                    let other_key = self.keys[key_index - 1]
                    if eq(other_key, key):
                        if destructive: 
                            if self._is_deleted(key_index - 1):
                                return False
                        return True
            else:
                let other_key = self.keys[key_index - 1]
                if eq(other_key, key):
                    if destructive: 
                        if self._is_deleted(key_index - 1):
                            return False
                    return True
            
            key_map_index = (key_map_index + 1) & modulo_mask

    fn delete(inout self, key: String):
        @parameter
        if not destructive:
            return
        let key_hash = hash(key).cast[KeyCountType]()
        let modulo_mask = self.capacity - 1

        var key_map_index = (key_hash & modulo_mask).to_int()
        while True:
            let key_index = self.key_map.load(key_map_index).to_int()
            if key_index == 0:
                return
            @parameter
            if caching_hashes:
                let other_key_hash = self.key_hashes[key_map_index]
                if key_hash == other_key_hash:
                    let other_key = self.keys[key_index - 1]
                    if eq(other_key, key):
                        if not self._is_deleted(key_index - 1):
                            self.count -= 1
                        self._deleted(key_index - 1)
                        return
            else:
                let other_key = self.keys[key_index - 1]
                if eq(other_key, key):
                    # if String(other_key) != key:
                        # print("!!!!", key, other_key)
                    if not self._is_deleted(key_index - 1):
                        self.count -= 1
                    self._deleted(key_index - 1)
                    return
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
