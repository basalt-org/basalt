from testing import assert_equal, assert_almost_equal
from basalt.nn import Tensor
from basalt import dtype

fn assert_tensors_equal[mode: String = "exact"](t1: Tensor[dtype], t2: Tensor[dtype]) raises:
    constrained[mode == "exact" or mode == "almost", "Mode must be either 'exact' or 'almost'"]()

    assert_equal(t1.shape(), t2.shape(), "Tensor shape mismatch")
    
    for i in range(t1.num_elements()):
        if mode == "almost":
            assert_almost_equal(t1[i], t2[i], rtol=1e-5, atol=1e-5)
        else:
            assert_equal(t1[i], t2[i])
