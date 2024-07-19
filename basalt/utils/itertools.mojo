
@value
struct _ProductIterator(Sized):
    var lists: List[List[Int]]
    var _current: Int
    var _iters: Int

    @always_inline("nodebug")
    fn __init__(inout self, lists: List[List[Int]]):
        self.lists = lists
        self._current = 0

        self._iters = 1
        for lst in self.lists:
            self._iters *= len(lst[])

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self._iters

    @always_inline("nodebug")
    fn __iter__(self) -> Self:
        return self

    @always_inline("nodebug")
    fn __next__(inout self) -> List[Int]:
        self._current += 1
        self._iters -= 1
        return self._get_combination(self._current - 1)

    @always_inline("nodebug")
    fn _get_combination(self, current: Int) -> List[Int]:
        var combination = List[Int]()
        var count = current
        for i in reversed(range(len(self.lists))):
            var index = count % len(self.lists[i])
            combination.append(self.lists[i][index])
            count //= len(self.lists[i])
        combination.reverse()
        return combination ^

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> List[Int]:
        return self._get_combination(index)


@always_inline("nodebug")
fn product(lists: List[List[Int]]) -> _ProductIterator:
    return _ProductIterator(lists)