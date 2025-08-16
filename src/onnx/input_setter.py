import onnx
from onnx import shape_inference
from onnx import TensorShapeProto
import argparse

def fix_symbolic_dimensions(model, input_shape):
    """
    Replace symbolic dimensions (like 'batch_size') with concrete values.
    This forces all tensors to have concrete shapes based on the input.
    """
    # Create mapping from symbolic names to concrete values
    # Assuming first dimension is batch size
    batch_size = input_shape[0]
    
    # Fix output tensor shapes
    for output_tensor in model.graph.output:
        for i, dim in enumerate(output_tensor.type.tensor_type.shape.dim):
            if dim.dim_param == "batch_size" or (dim.dim_param and i == 0):
                dim.dim_value = batch_size
                dim.ClearField('dim_param')
    
    # Fix intermediate tensor shapes
    for value_info in model.graph.value_info:
        for i, dim in enumerate(value_info.type.tensor_type.shape.dim):
            if dim.dim_param == "batch_size" or (dim.dim_param and i == 0):
                dim.dim_value = batch_size
                dim.ClearField('dim_param')
    
    return model

def set_input_shape(model, input_shape):
    """
    Set the input tensor shape for the first input of an ONNX model.
    Can handle changing the number of dimensions.
    
    Parameters:
    model (onnx.ModelProto): The loaded ONNX model object.
    input_shape (list[int]): A list of integers specifying the new input shape.
    
    Returns:
    onnx.ModelProto: The model with the updated input shape.
    """
    # Get the first input tensor
    input_tensor = model.graph.input[0]
    
    # Clear existing dimensions
    input_tensor.type.tensor_type.shape.ClearField('dim')
    
    # Add new dimensions
    for dim_value in input_shape:
        new_dim = input_tensor.type.tensor_type.shape.dim.add()
        new_dim.dim_value = dim_value
    
    print(f"Updated input '{input_tensor.name}' shape to: {input_shape}")
    return model

def print_model_info(model):
    """Print information about the model inputs and outputs"""
    print("\n=== Model Information ===")
    
    print("\nInputs:")
    for i, input_tensor in enumerate(model.graph.input):
        name = input_tensor.name
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  {i}: {name} -> {shape}")
    
    print("\nOutputs:")
    for i, output_tensor in enumerate(model.graph.output):
        name = output_tensor.name
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  {i}: {name} -> {shape}")

def main():
    """
    Command-line utility to update the input shape of an ONNX model and
    run ONNX's shape inference to deduce shapes of all intermediate tensors.
    
    Workflow:
    1. Parse command-line arguments: path to ONNX model and desired input shape.
    2. Load the ONNX model from the provided path.
    3. Set the model's input shape to the specified values.
    4. Run ONNX shape inference to automatically compute shapes of all tensors.
    5. Print out the names and shapes of all inferred tensors.
    6. Save the updated model back to the original file path.
    """
    parser = argparse.ArgumentParser(description="Set input shape and infer shapes of the ONNX model")
    parser.add_argument("--path", type=str, required=True, help="Path of your model.")
    parser.add_argument("--shape", type=str, required=True, 
                       help="Input shape as comma-separated values, e.g., 1,3,34,34 or 1,1,1,5")
    
    args = parser.parse_args()
    
    # Parse the shape argument
    input_shape = list(map(int, args.shape.split(",")))
    print(f"Target input shape: {input_shape}")
    
    # Load the ONNX model from the given file path
    print(f"Loading model from: {args.path}")
    model = onnx.load(args.path)
    
    # Print current model info
    print("\n=== BEFORE MODIFICATION ===")
    print_model_info(model)
    
    # Update the model's first input tensor with the specified shape
    model = set_input_shape(model, input_shape)
    
    try:
        # Perform shape inference to fill in tensor shapes for all intermediate nodes
        print("\nRunning shape inference...")
        inferred_model = shape_inference.infer_shapes(model)
        
        # Fix symbolic dimensions to concrete values
        print("Fixing symbolic dimensions to concrete values...")
        inferred_model = fix_symbolic_dimensions(inferred_model, input_shape)
        
        # Print updated model info
        print("\n=== AFTER MODIFICATION ===")
        print_model_info(inferred_model)
        
        # Print out the names and shapes of all inferred tensors for verification
        print("\n=== All Inferred Tensor Shapes ===")
        for value_info in inferred_model.graph.value_info:
            name = value_info.name
            shape = [dim.dim_value if dim.dim_value > 0 else str(dim.dim_param) 
                    for dim in value_info.type.tensor_type.shape.dim]
            print(f"  {name}: {shape}")
        
        # Save the updated model (with inferred shapes) back to the original path
        print(f"\nSaving updated model to: {args.path}")
        onnx.save(inferred_model, args.path)
        print("✅ Model updated successfully!")
        
    except Exception as e:
        print(f"⚠️  Shape inference failed: {e}")
        print("Saving model without shape inference...")
        onnx.save(model, args.path)
        print("✅ Model saved (without complete shape inference)")

if __name__ == "__main__":
    main()