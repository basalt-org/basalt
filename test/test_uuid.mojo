from testing import assert_not_equal, assert_equal, assert_true, assert_false

from dainemo import seed
from dainemo.utils.uuid import UUID, ID


fn test_uuid_length() raises:
    var uuid = UUID(seed)
    var id = uuid.next()

    var splitted = str(id).split("-")
    var char_count = 0
    for i in range(splitted.size):
        char_count += len(splitted[i])

    assert_equal(char_count, 32)
    assert_equal(len(str(id)), 32+4)


fn dv_contains(dv: DynamicVector[ID], symbol: ID) -> Bool:
    for i in range(len(dv)):
        if dv[i] == symbol:
            return True
    return False


fn test_uuid_uniqueness() raises:
    var uuid = UUID(seed)
    var seen = DynamicVector[ID]()
    alias N = 10_000 #1_000_000
    for i in range(N):
        # print(i)
        var id = uuid.next()
        assert_false(dv_contains(seen, id))
        seen.push_back(id)


fn test_uuid_version() raises:
    var uuid = UUID(seed)
    for i in range(10):
        var id = uuid.next()
        assert_equal(str(id).split("-")[2][0], '4')


fn test_uuid_variant() raises:
    var uuid = UUID(seed)
    for i in range(10):
        var id = uuid.next()
        var variant = str(id).split("-")[3][0]
        var variant_condition = variant == '8' or variant == '9' or variant == 'a' or variant == 'b'
        assert_true(variant_condition)


fn main():
    try:
        test_uuid_length()
        test_uuid_uniqueness()
        test_uuid_version()
        test_uuid_variant()      
    except:
        print("[ERROR] Error in test_uuid.py")
