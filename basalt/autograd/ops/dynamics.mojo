from basalt import Symbol
from basalt import TENSORS, GRADS
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
    fn forward[attributes: AttributeVector,](
        inputs: List[Symbol],
        outputs: List[Symbol]
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
                    TENSORS[outputs[0]].data() + i * chunk_offsets[len(inputs)] + chunk_offsets[j],
                    TENSORS[inputs[j]].data() + i * chunks[j],
                    chunks[j],
                )

    @staticmethod
    fn backward[input_id: Int, attributes: AttributeVector](
        inputs: List[Symbol],
        outputs: List[Symbol]
    ) -> Tensor[dtype]:
        alias dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        # TODO
        return Tensor[dtype]()
        
        # alias n_chunks = Self.calc_chunks(t1_shape, dim)
        # alias chunk_1 = t1_shape.num_elements() // n_chunks
        # alias chunk_2 = t2_shape.num_elements() // n_chunks

        # @parameter
        # if tensor_id == 0:
        #     var t1_size = t1_shape[dim]
        #     var t1_grad = Tensor[dtype](t1_shape)
            
        #     @unroll
        #     for i in range(n_chunks):
        #         memcpy(
        #             t1_grad.data() + i * chunk_1,
        #             ug.data() + i * (chunk_1 + chunk_2),
        #             chunk_1,
        #         )

        #     return t1_grad ^
        # else:
        #     var t2_size = t2_shape[dim]
        #     var t2_grad = Tensor[dtype](t2_shape)
            
        #     @unroll
        #     for i in range(n_chunks):
        #         memcpy(
        #             t2_grad.data() + i * chunk_2,
        #             ug.data() + i * (chunk_1 + chunk_2) + chunk_1,
        #             chunk_2,
        #         )

        #     return t2_grad ^


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
    fn forward[attributes: AttributeVector,](
        inputs: List[Symbol],
        outputs: List[Symbol]
    ):
        alias dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        alias sections = attributes["sections"].value().to_shape()
        # TODO
        pass

    @staticmethod
    fn backward[input_id: Int, attributes: AttributeVector](
        inputs: List[Symbol],
        outputs: List[Symbol]
    ) -> Tensor[dtype]:
        alias dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        alias sections = attributes["sections"].value().to_shape()
        # TODO
        return Tensor[dtype]()