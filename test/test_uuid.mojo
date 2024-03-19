from time import now
from testing import assert_equal, assert_true, assert_false

from basalt.utils.uuid import UUIDGenerator, UUID


alias seed = 42


fn test_uuid_length() raises:
    var uuid_generator = UUIDGenerator(seed)
    var uuid = uuid_generator.next()

    var splitted = str(uuid).split("-")
    var char_count = 0
    for i in range(splitted.size):
        char_count += len(splitted[i])

    assert_equal(char_count, 32)
    assert_equal(len(str(uuid)), 32+4)


fn test_uuid_version() raises:
    var uuid_generator = UUIDGenerator(seed)
    for i in range(10):
        var uuid = uuid_generator.next()
        assert_equal(str(uuid).split("-")[2][0], '4')


fn test_uuid_variant() raises:
    var uuid_generator = UUIDGenerator(seed)
    for i in range(10):
        var uuid = uuid_generator.next()
        var variant = str(uuid).split("-")[3][0]
        var variant_condition = variant == '8' or variant == '9' or variant == 'a' or variant == 'b'
        assert_true(variant_condition, "Variant is not 8, 9, a or b")


fn dv_contains(dv: DynamicVector[UUID], uuid: UUID) -> Bool:
    for i in range(len(dv)):
        if dv[i] == uuid:
            return True
    return False


fn test_uuid_uniqueness() raises:
    var uuid_generator = UUIDGenerator(seed)
    var seen = DynamicVector[UUID]()
    
    alias N = 100_000 #1_000_000
    var start = now()
    for i in range(N):
        var uuid = uuid_generator.next()
        assert_false(dv_contains(seen, uuid), "UUID is not unique")
        seen.push_back(uuid)
    #     if i % 1000 == 0:
    #         print("Progress: ", i, "/", N)
    # print("Time: ", (now() - start)/1e9, "s")


fn test_uuid_compile_time() raises:
    
    fn generate_uuid() -> UUID:
        var uuid_generator = UUIDGenerator(seed)
        return uuid_generator.next()
    
    alias uuid = generate_uuid()



fn main():
    
    try:
        test_uuid_length()
        test_uuid_version()
        test_uuid_variant()
        test_uuid_uniqueness()
        test_uuid_compile_time()
    except e:
        print("[ERROR] Error in test_uuid: ")
        print(e)
