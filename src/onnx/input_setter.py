import onnx
from onnx import shape_inference
import argparse

def set_input_shape(model, input_shape):
    # Ensure the input has the expected number of dimensions
    if len(input_shape) != 4:
        raise ValueError("Input shape must be a list of 4 integers: [N, C, H, W]")

    for i in range(4):
        model.graph.input[0].type.tensor_type.shape.dim[i].dim_value = input_shape[i]

    return model

def main():

    parser = argparse.ArgumentParser(description=" Set input shape and infer shapes of the ONNX model")
    parser.add_argument("--path", type=str, required=True, help="Path of your model.")
    parser.add_argument("--shape", type=str, required=True, help="Input shape as comma-separated values, e.g., 1,3,34,34")
    args = parser.parse_args()
    
     # Parse shape argument
    input_shape = list(map(int, args.shape.split(",")))
    if len(input_shape) != 4:
        raise ValueError("Input shape must be in the format N,C,H,W (e.g., 1,3,34,34)")

    # Load and process the model
    model = onnx.load(args.path)
    model = set_input_shape(model, input_shape)
    # Infer shapes
    inferred_model = shape_inference.infer_shapes(model)

    for value_info in inferred_model.graph.value_info:
        name = value_info.name
        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        print(f"{name}: {shape}")

    onnx.save(inferred_model,  args.path)

if __name__ == "__main__":
    main()
