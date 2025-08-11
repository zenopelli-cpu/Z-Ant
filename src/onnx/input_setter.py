import onnx
from onnx import shape_inference
import argparse

def set_input_shape(model, input_shape):
    """
    Set the input tensor shape for the first input of an ONNX model.

    Parameters:
        model (onnx.ModelProto): The loaded ONNX model object.
        input_shape (list[int]): A list of exactly four integers specifying 
                                 the input shape in NCHW format:
                                 N = batch size
                                 C = number of channels
                                 H = height
                                 W = width

    Returns:
        onnx.ModelProto: The model with the updated input shape.

    Raises:
        ValueError: If the provided input shape does not have exactly four elements.
    """
    # Ensure the input has the expected number of dimensions
    if len(input_shape) != 4:
        raise ValueError("Input shape must be a list of 4 integers: [N, C, H, W]")

    # Assign the given dimensions to the first input tensor in the model
    for i in range(4):
        model.graph.input[0].type.tensor_type.shape.dim[i].dim_value = input_shape[i]

    return model

def main():
    """
    Command-line utility to update the input shape of an ONNX model and 
    run ONNX's shape inference to deduce shapes of all intermediate tensors.

    Workflow:
        1. Parse command-line arguments: path to ONNX model and desired input shape.
        2. Load the ONNX model from the provided path.
        3. Set the model's input shape to the specified NCHW values.
        4. Run ONNX shape inference to automatically compute shapes of all tensors.
        5. Print out the names and shapes of all inferred tensors.
        6. Save the updated model back to the original file path.
    """
    parser = argparse.ArgumentParser(description="Set input shape and infer shapes of the ONNX model")
    parser.add_argument("--path", type=str, required=True, help="Path of your model.")
    parser.add_argument("--shape", type=str, required=True, help="Input shape as comma-separated values, e.g., 1,3,34,34")
    args = parser.parse_args()
    
    # Parse and validate the shape argument
    input_shape = list(map(int, args.shape.split(",")))
    if len(input_shape) != 4:
        raise ValueError("Input shape must be in the format N,C,H,W (e.g., 1,3,34,34)")

    # Load the ONNX model from the given file path
    model = onnx.load(args.path)

    # Update the model's first input tensor with the specified shape
    model = set_input_shape(model, input_shape)

    # Perform shape inference to fill in tensor shapes for all intermediate nodes
    inferred_model = shape_inference.infer_shapes(model)

    # Print out the names and shapes of all inferred tensors for verification
    for value_info in inferred_model.graph.value_info:
        name = value_info.name
        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        print(f"{name}: {shape}")

    # Save the updated model (with inferred shapes) back to the original path
    onnx.save(inferred_model, args.path)

if __name__ == "__main__":
    main()
