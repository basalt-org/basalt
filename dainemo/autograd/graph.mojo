from tensor import Tensor


struct Graph:
    '''
    Keep track of all nodes in the computational graph.
    Created during the forward pass.
    Used by the backpard pass to compute gradients with autodiff.
    '''

    pass