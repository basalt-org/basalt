import onnx
from onnx import helper
from onnx import TensorProto
import netron


def get_param_data(param_shape):
    factor = 1
    for dim in param_shape:
        factor *= dim
    return [0] * factor


def create_onnx_graph_from_json(graph, type='node'):
        
    # Create a list to hold nodes, inputs, outputs, and initializers
    nodes = []
    inputs = []
    outputs = []
    initializers = []
    intermediates = []
    
    # Process params as initializers (if operator-graph)
    visited = []
    if type == 'operator':
        onnx_inputs = graph['inputs'] + graph.get('params', [])
    elif type == 'node':
        onnx_inputs = graph['inputs']

        # Process params as initializers
        for initializer in graph.get('params', []):
            name = initializer['name']
            dtype = TensorProto.FLOAT # TODO
            shape = list(map(int, initializer['shape'].split("x")))
            tensor = helper.make_tensor(name, dtype, shape, get_param_data(shape))
            initializers.append(tensor)
            visited.append(name)

    # Process inputs
    for input in onnx_inputs:
        name = input['name']
        dtype = TensorProto.FLOAT # TODO
        shape = list(map(int, input['shape'].split("x")))
        inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        visited.append(name)

    # Process outputs
    for output in graph['outputs']:
        name = output['name']
        dtype = TensorProto.FLOAT # TODO
        shape = list(map(int, output['shape'].split("x")))
        outputs.append(helper.make_tensor_value_info(name, dtype, shape))
        visited.append(name)

    # Process nodes
    for node in graph['nodes']:
        operator = node['operator']
        onnx_node = helper.make_node(
            operator,
            inputs=[input['name'] for input in node['inputs']],
            outputs=[output['name'] for output in node['outputs']],
            name=f"{node['operator']}_node"
        )
        nodes.append(onnx_node)

        # Process intermediates
        for output in node['outputs']:
            if output['name'] not in visited:
                name = output['name']
                dtype = TensorProto.FLOAT
                shape = list(map(int, output['shape'].split("x")))
                intermediates.append(helper.make_tensor_value_info(name, dtype, shape))
                visited.append(name)    

    # Process loss
    if 'loss' in graph.keys():
        loss = graph['loss'][0]
        name = loss['name']
        dtype = TensorProto.FLOAT
        shape = list(map(int, loss['shape'].split("x")))
        outputs.append(helper.make_tensor_value_info(name, dtype, shape))
        visited.append(name)

    # Create the graph
    graph_def = helper.make_graph(
        nodes,
        graph.get('graph_name', 'basalt-ONNX'),
        inputs,
        outputs,
        initializer=initializers,
        value_info=intermediates
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name='basalt')

    # Save the model to a file
    onnx.save(model_def, "output_model.onnx")


def netron_render(graph, type='node'):
    assert type in ['node', 'operator']
    create_onnx_graph_from_json(graph, type=type)
    netron.start('output_model.onnx')