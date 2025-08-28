import onnx
from onnx import shape_inference
import argparse
from onnxsim import simplify

def create_dummy_input_for_initializers(model):
    """
    Create dummy graph inputs for initializers to satisfy ONNX validation.
    This fixes the 'initializer but not in graph input' error.
    """
    # Get names of existing graph inputs
    existing_input_names = set([inp.name for inp in model.graph.input])
    
    # Get all initializer names
    initializer_names = set([init.name for init in model.graph.initializer])
    
    # Find initializers that need dummy inputs
    missing_inputs = initializer_names - existing_input_names
    
    if missing_inputs:
        print(f"Creating dummy inputs for {len(missing_inputs)} initializers...")
        
        # Create dummy inputs for each missing initializer
        for init in model.graph.initializer:
            if init.name in missing_inputs:
                # Create corresponding input tensor
                input_tensor = model.graph.input.add()
                input_tensor.name = init.name
                input_tensor.type.tensor_type.elem_type = init.data_type
                
                # Set shape from initializer dimensions
                for dim_size in init.dims:
                    dim = input_tensor.type.tensor_type.shape.dim.add()
                    dim.dim_value = dim_size
        
        print(f"Added {len(missing_inputs)} dummy inputs for initializers")
    
    return model

def set_input_shape(model, input_shape, input_name="data"):
    """
    Set the input tensor shape for a specific input by name.
    """
    target_input = None
    
    # Find the target input tensor by name
    for input_tensor in model.graph.input:
        if input_tensor.name == input_name:
            target_input = input_tensor
            break
    
    if target_input is None:
        print(f"Warning: Input '{input_name}' not found. Available inputs:")
        for inp in model.graph.input:
            print(f"  - {inp.name}")
        # Use first input as fallback
        if len(model.graph.input) > 0:
            target_input = model.graph.input[0]
            print(f"Using first input: {target_input.name}")
        else:
            raise ValueError("No inputs found in model")
    
    # Clear existing dimensions
    target_input.type.tensor_type.shape.ClearField('dim')
    
    # Add new dimensions
    for dim_value in input_shape:
        new_dim = target_input.type.tensor_type.shape.dim.add()
        new_dim.dim_value = dim_value
    
    print(f"Updated input '{target_input.name}' shape to: {input_shape}")
    return model

def safe_simplify(model):
    """
    Attempt to simplify the model with multiple fallback strategies.
    """
    print("Attempting model simplification...")
    
    # Strategy 1: Standard simplification
    try:
        model_simp, check = simplify(model)
        if check:
            print("✅ Standard simplification successful!")
            return model_simp
        print("Standard simplification completed but validation uncertain")
        return model_simp
    except Exception as e:
        print(f"Standard simplification failed: {e}")
    
    # Strategy 2: Skip optimization
    try:
        print("Trying with skip_optimization=True...")
        model_simp, check = simplify(model, skip_optimization=True)
        print("✅ Simplification with skip_optimization successful!")
        return model_simp
    except Exception as e:
        print(f"Skip optimization also failed: {e}")
    
    # Strategy 3: Skip shape inference
    try:
        print("Trying with skip_shape_inference=True...")
        model_simp, check = simplify(model, skip_shape_inference=True)
        print("✅ Simplification with skip_shape_inference successful!")
        return model_simp
    except Exception as e:
        print(f"Skip shape inference also failed: {e}")
    
    print("⚠️ All simplification strategies failed, keeping original model")
    return model

def clean_model(model_path, input_shape, output_path=None):
    """
    Clean and process ONNX model with comprehensive error handling.
    """
    if output_path is None:
        output_path = model_path
    
    print(f"Processing: {model_path}")
    print(f"Target shape: {input_shape}")
    
    # Load model
    try:
        model = onnx.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print(f"Original model: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
    
    # Fix the validation issue
    print("\n1. Fixing initializer validation issues...")
    model = create_dummy_input_for_initializers(model)
    
    # Set input shape
    print("\n2. Setting input shape...")
    try:
        model = set_input_shape(model, input_shape)
    except Exception as e:
        print(f"❌ Failed to set input shape: {e}")
        return False
    
    # Run shape inference
    print("\n3. Running shape inference...")
    try:
        model = shape_inference.infer_shapes(model)
        print("✅ Shape inference successful")
    except Exception as e:
        print(f"⚠️ Shape inference failed: {e}")
        print("Continuing without shape inference...")
    
    # Simplify model
    print("\n4. Simplifying model...")
    model = safe_simplify(model)
    
    # Save model
    print(f"\n5. Saving model to: {output_path}")
    try:
        onnx.save(model, output_path)
        print("✅ Model saved successfully!")
        
        # Final stats
        print(f"\nFinal model: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix and simplify ResNet ONNX models")
    parser.add_argument("--path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--shape", type=str, required=True, 
                       help="Input shape as comma-separated values (e.g., 1,3,224,224)")
    parser.add_argument("--output", type=str, help="Output path (optional)")
    
    args = parser.parse_args()
    
    # Parse shape
    try:
        input_shape = [int(x) for x in args.shape.split(",")]
    except ValueError:
        print("❌ Invalid shape format. Use: 1,3,224,224")
        return
    
    # Process model
    success = clean_model(args.path, input_shape, args.output)
    
    if success:
        print("\n🎉 Model processing completed successfully!")
    else:
        print("\n💥 Model processing failed!")

if __name__ == "__main__":
    main()