from basalt import Symbol
from basalt.nn.model import Parameters
from ..attributes import AttributeVector


struct CONCAT:
    @staticmethod
    fn result_shape(
        input_shapes: List[TensorShape], attributes: AttributeVector
    ) -> List[TensorShape]:
        # Assumptions: all tensors have the same shape, except for the concatenating dimension
        var dim = attributes["dim"].value().to_int() if attributes["dim"] else 0

        var concat_size: Int = 0
        for i in range(len(input_shapes)):
            concat_size += input_shapes[i][dim]

        var res_shape = input_shapes[0]
        res_shape[dim] = concat_size

        return List[TensorShape](res_shape)

    @staticmethod
    fn calc_chunks(shape: TensorShape, dim: Int) -> Int:
        # Number of chunks up to the concatenating dimension
        # Assuming tensor of equal shape, except for the concatenating dimension
        var chunks = 1
        for i in range(dim):
            chunks *= shape[i]
        return chunks

    @staticmethod
    fn forward[
        attributes: AttributeVector,
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        parameters: Reference[Parameters, mutability, lifetime],
    ):
        alias dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets = List[Int](0)
        for i in range(len(inputs)):
            chunks.append(inputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        for i in range(n_chunks):
            for j in range(len(inputs)):
                memcpy(
                    parameters[].tensors[outputs[0]].data()
                    + i * chunk_offsets[len(inputs)]
                    + chunk_offsets[j],
                    parameters[].tensors[inputs[j]].data() + i * chunks[j],
                    chunks[j],
                )

    @staticmethod
    fn backward[
        input_id: Int,
        attributes: AttributeVector,
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        parameters: Reference[Parameters, mutability, lifetime],
    ) -> Tensor[dtype]:
        alias dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets = List[Int](0)
        for i in range(len(inputs)):
            chunks.append(inputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        var res_grad = Tensor[dtype](inputs[input_id].shape)
        for i in range(n_chunks):
            memcpy(
                res_grad.data() + i * chunks[input_id],
                parameters[].grads[outputs[0]].data()
                + i * chunk_offsets[len(inputs)]
                + chunk_offsets[input_id],
                chunks[input_id],
            )

        return res_grad ^


struct SPLIT:
    @staticmethod
    fn result_shape(
        input_shapes: List[TensorShape], attributes: AttributeVector
    ) -> List[TensorShape]:
        # Assuming the sum of the sections is equal to the total size in the dim dimension.
        # E.g. sections = [5, 5, 2] -> shape (., 12, ., .) for dim = 1
        var dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        var sections = attributes["sections"].value().to_shape()

        var res_shapes = List[TensorShape]()
        for i in range(sections.rank()):
            var new_shape = input_shapes[0]
            new_shape[dim] = sections[i]
            res_shapes.append(new_shape)

        return res_shapes

    @staticmethod
    fn calc_chunks(shape: TensorShape, dim: Int) -> Int:
        # Number of chunks up to the concatenating dimension
        # Assuming tensor of equal shape, except for the concatenating dimension
        var chunks = 1
        for i in range(dim):
            chunks *= shape[i]
        return chunks

    @staticmethod
    fn forward[
        attributes: AttributeVector,
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        parameters: Reference[Parameters, mutability, lifetime],
    ):
        alias dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        alias sections = attributes["sections"].value().to_shape()
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets = List[Int](0)
        for i in range(len(outputs)):
            chunks.append(outputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        for i in range(n_chunks):
            for j in range(len(outputs)):
                memcpy(
                    parameters[].tensors[outputs[j]].data() + i * chunks[j],
                    parameters[].tensors[inputs[0]].data()
                    + i * chunk_offsets[len(outputs)]
                    + chunk_offsets[j],
                    chunks[j],
                )

    @staticmethod
    fn backward[
        input_id: Int,
        attributes: AttributeVector,
        mutability: __mlir_type.i1,
        lifetime: AnyLifetime[mutability].type,
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        parameters: Reference[Parameters, mutability, lifetime],
    ) -> Tensor[dtype]:
        alias dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        alias sections = attributes["sections"].value().to_shape()
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets = List[Int](0)
        for i in range(len(outputs)):
            chunks.append(outputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        var res_grad = Tensor[dtype](inputs[input_id].shape)

        for i in range(n_chunks):
            for j in range(len(outputs)):
                memcpy(
                    res_grad.data()
                    + i * chunk_offsets[len(outputs)]
                    + chunk_offsets[j],
                    parameters[].grads[outputs[j]].data() + i * chunks[j],
                    chunks[j],
                )

        return res_grad ^
