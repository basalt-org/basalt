from testing import assert_not_equal, assert_equal, assert_true
from dainemo.utils.uuid import uuid


fn test_uuid_length() raises:
    var id = uuid()

    var splitted = id.split("-")
    var char_count = 0
    for i in range(splitted.size):
        char_count += len(splitted[i])

    assert_equal(char_count, 32)
    assert_equal(len(id), 32+4)


fn test_uuid_uniqueness() raises:
    for i in range(10):
        var id1 = uuid()
        var id2 = uuid()
        assert_not_equal(id1, id2)


fn test_uuid_version() raises:
    for i in range(10):
        var id = uuid()
        assert_equal(id.split("-")[2][0], '4')


fn test_uuid_variant() raises:
    for i in range(10):
        var id = uuid()
        var variant = id.split("-")[3][0]
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
