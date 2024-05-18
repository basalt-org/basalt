from algorithm import parallelize as builtin_parallelize
from sys.info import (
    num_physical_cores,
    is_apple_silicon,
    has_intel_amx,
    has_avx512f,
    has_neon,
)

@always_inline("nodebug")
fn num_physical_cores() -> Int:
    # return num_physical_cores
    return 16

@always_inline("nodebug")
fn should_parallelize() -> Bool:
    return (
        is_apple_silicon()
        or has_intel_amx()
        or has_avx512f()
        or has_neon()
        or num_physical_cores() >= 4
    )


@always_inline("nodebug")
fn parallelize[func: fn (Int, /) capturing -> None](num_work_items: Int):
    @parameter
    if should_parallelize():
        builtin_parallelize[func](num_work_items)
    else:
        sequential[func](num_work_items)


@always_inline("nodebug")
fn parallelize[
    func: fn (Int, /) capturing -> None
](num_work_items: Int, num_workers: Int):
    @parameter
    if should_parallelize():
        builtin_parallelize[func](num_work_items, num_workers)
    else:
        sequential[func](num_work_items)


@always_inline("nodebug")
fn sequential[func: fn (Int, /) capturing -> None](num_work_items: Int):
    for i in range(num_work_items):
        func(i)
