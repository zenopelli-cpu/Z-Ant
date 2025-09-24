import numpy as np
import random
from onnx import helper, TensorProto


def generate_onehot_model(input_names, output_names):
    """
    Generates a OneHot operator model.
    """
    initializers = []
    
    # Operatore OneHot: genera indices, depth e values come initializer
    # Genera un tensore indices con rango casuale (1 o 2)
    rank = random.randint(1, 2)
    indices_shape = [random.randint(2, 3) for _ in range(rank)]
    max_index = 3  # Limite massimo per gli indici
    indices_data = np.random.randint(0, max_index, size=indices_shape).astype(np.int64)
    indices_tensor = helper.make_tensor(input_names[0], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
    initializers.append(indices_tensor)

    # Genera depth (scalare)
    depth_value = random.randint(3, max_index)  # Valore positivo per depth
    depth_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [], [depth_value])
    initializers.append(depth_tensor)

    # Genera values (tensore 1D di lunghezza 2, tipo float32)
    values_data = np.array([0.0, 1.0], dtype=np.float32)  # [off_value, on_value]
    values_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [2], values_data.tolist())
    initializers.append(values_tensor)

    # Scegli un axis valido
    output_rank = rank + 1
    axis = random.randint(max(-output_rank, -1000), min(output_rank - 1, 1000))

    # Calcola la forma dell'output
    out_shape = indices_shape.copy()
    normalized_axis = axis if axis >= 0 else axis + output_rank
    out_shape.insert(normalized_axis, depth_value)

    # Crea il nodo OneHot
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
    node = helper.make_node(
        "OneHot",
        inputs=[input_names[0], input_names[1], input_names[2]],
        outputs=[output_names[0]],
        axis=axis,
        name=f"OneHot_node_axis{axis}"
    )

    # Input info fittizio
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, indices_shape)

    # Metadati
    metadata = {
        "input_shapes": [indices_shape, [], [2]],  # Forme di indices, depth, values
        "output_shapes": [out_shape],
        "axis": axis,
        "depth": depth_value,
        "indices": indices_data.flatten().tolist()[:5],  # Solo i primi 5 per debug
        "values": values_data.tolist()
    }

    return [input_info], output_info, [node], initializers, metadata