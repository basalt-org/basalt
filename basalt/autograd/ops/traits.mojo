from basalt.autograd.attributes import AttributeVector

trait Operator:
    ...


trait UnaryOperator(Operator):
    @staticmethod
    fn result_shape(
        input_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
        """
        Computes the result shape of an operation given the input shape and attributes.
        Runs at compile time.
        """
        ...

    @staticmethod
    fn forward[InputShape: TensorShape, Attributes: AttributeVector](inout res: Tensor[dtype], input: Tensor[dtype]):
        """
        Computes the forward pass of an operation given the input tensor.
        """
        ...

    @staticmethod
    fn backward[TensorID: Int, GradShape: TensorShape, InputShape: TensorShape](
        grad: Tensor[dtype],
        input: Tensor[dtype],
    ) -> Tensor[dtype]:
        """
        Computes the backward pass of an operation given the gradient tensor.
        """
        ...


trait BinaryOperator(Operator):
    @staticmethod
    fn result_shape(
        lhs_shape: TensorShape, rhs_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
        """
        Computes the result shape of an operation given the input shapes and attributes.
        Runs at compile time.
        """
        ...

    @staticmethod
    fn forward[LeftShape: TensorShape, RightShape: TensorShape, Attributes: AttributeVector](
        inout res: Tensor[dtype], lhs: Tensor[dtype], rhs: Tensor[dtype]):
        """
        Computes the forward pass of an operation given the input tensors.
        """
        ...

    @staticmethod
    fn backward[
        TensorID: Int,
        GradShape: TensorShape,
        LeftShape: TensorShape,
        RightShape: TensorShape,
    ](
        grad: Tensor[dtype],
        lhs: Tensor[dtype],
        rhs: Tensor[dtype],
    ) -> Tensor[dtype]:
        """
        Computes the backward pass of an operation given the gradient tensor.
        """
        ...


trait TernaryOperator(Operator):
    @staticmethod
    fn result_shape(
        lhs_shape: TensorShape,
        mid_shape: TensorShape,
        rhs_shape: TensorShape,
        attributes: AttributeVector,
    ) -> TensorShape:
        """
        Computes the result shape of an operation given the input shapes and attributes.
        Runs at compile time.
        """
        ...

    @staticmethod
    fn forward[
        LeftShape: TensorShape,
        MidShape: TensorShape,
        RightShape: TensorShape,
        Attributes: AttributeVector,
    ](
        inout res: Tensor[dtype],
        lhs: Tensor[dtype],
        mid: Tensor[dtype],
        rhs: Tensor[dtype],
    ):
        """
        Computes the forward pass of an operation given the input tensors.
        """
        ...

    @staticmethod
    fn backward[
        TensorID: Int,
        GradShape: TensorShape,
        LeftShape: TensorShape,
        MidShape: TensorShape,
        RightShape: TensorShape,
    ](
        grad: Tensor[dtype],
        lhs: Tensor[dtype],
        mid: Tensor[dtype],
        rhs: Tensor[dtype],
    ) -> Tensor[dtype]:
        """
        Computes the backward pass of an operation given the gradient tensor.
        """
        ...
