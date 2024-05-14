
@value
struct _ProductIterator(Sized):
    var lists: List[List[Int]]
    var indeces: List[Int]
    var _iters: Int

    @always_inline("nodebug")
    fn __init__(inout self, lists: List[List[Int]]):
        self.lists = lists
        self.indeces = List[Int]()
        for i in range(len(lists)):
            self.indeces.append(0)
        
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
        var res = List[Int]()
        for i in range(len(self.lists)):
            res.append(self.lists[i][self.indeces[i]])
        self._increment_indeces()
        self._iters -= 1
        return res ^

    @always_inline("nodebug")
    fn _increment_indeces(inout self):
        for i in reversed(range(len(self.indeces))):
            self.indeces[i] += 1
            if self.indeces[i] < len(self.lists[i]):
                break
            self.indeces[i] = 0


@always_inline("nodebug")
fn product(lists: List[List[Int]]) -> _ProductIterator:
    return _ProductIterator(lists)