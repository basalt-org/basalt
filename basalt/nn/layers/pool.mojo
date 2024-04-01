from collections.optional import Optional

from basalt.autograd.attributes import AttributeVector, Attribute


@always_inline("nodebug")
fn set_static_stride(
    kernel_size: StaticIntTuple[2], stride: Optional[Int] = None
) -> StaticIntTuple[2]:
    if stride:
        return StaticIntTuple[2](stride.value(), stride.value())
    else:
        return kernel_size


@always_inline("nodebug")
fn MaxPool2d(
    inout g: Graph,
    inputs: Symbol,
    kernel_size: StaticIntTuple[2],
    stride: Optional[Int] = None,
    padding: StaticIntTuple[2] = 0,
    dilation: StaticIntTuple[2] = 1,
) -> Symbol:
    """
    A 2D Max Pooling Layer.

    Kernel is unaware of the in_channels and out_channels of the input tensor.
    kernel.size     (kX, kY)
    """

    # TODO: assert padding <= kernel_size / 2 (at compile time)

    var stride_temp = set_static_stride(kernel_size, stride)

    return MaxPool2d(g, inputs, kernel_size, stride_temp, padding, dilation)


@always_inline("nodebug")
fn MaxPool2d(
    inout g: Graph,
    inputs: Symbol,
    kernel_size: StaticIntTuple[2],
    stride: StaticIntTuple[2],  # stride should be 1 or more
    padding: StaticIntTuple[2] = 0,
    dilation: StaticIntTuple[2] = 1,
) -> Symbol:
    """
    A 2D Max Pooling Layer.

    Kernel is unaware of the in_channels and out_channels of the input tensor.
    kernel.size     (kX, kY)
    """
    # TODO: assert padding <= kernel_size / 2 (at compile time)

    return g.op(
        OP.MAXPOOL2D,
        inputs,
        attributes=AttributeVector(
            Attribute("kernel_size", kernel_size),
            Attribute("padding", padding),
            Attribute("stride", stride),
            Attribute("dilation", dilation),
        ),
    )


# # TODO
