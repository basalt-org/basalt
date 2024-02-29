# This code is based on https://github.com/mzaks/compact-dict

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
