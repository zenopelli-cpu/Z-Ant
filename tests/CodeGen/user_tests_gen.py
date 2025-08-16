import onnx
import onnxruntime as ort
import argparse
import json
import numpy as np
import os
import traceback

def get_runtime_input_shapes(model_path):
    """Get only the runtime inputs (not parameters/weights)"""
    sess = ort.InferenceSession(model_path)
    input_shapes = {}
    
    # Get actual runtime inputs from the session
    for input_meta in sess.get_inputs():
        name = input_meta.name
        shape = input_meta.shape
        # Replace dynamic dimensions with 1
        dims = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
        input_shapes[name] = dims
        
    return input_shapes

def generate_random_input(shape, input_name=""):
    """Generate random input data"""
    if "mnist" in input_name.lower() or any(dim == 28 for dim in shape):
        # For MNIST-like inputs, generate values in [0, 1] range
        return np.random.rand(*shape).astype(np.float32)
    else:
        # For other inputs, use standard normal distribution
        return np.random.randn(*shape).astype(np.float32)

def run_onnx_inference(model_path, input_shapes):
    """Run inference with the ONNX model"""
    sess = ort.InferenceSession(model_path)
    
    # Generate inputs only for runtime inputs
    inputs = {}
    for name, shape in input_shapes.items():
        inputs[name] = generate_random_input(shape, name)
    
    # Run inference
    outputs = sess.run(None, inputs)
    output_names = [output.name for output in sess.get_outputs()]
    
    return inputs, dict(zip(output_names, outputs))

def main():
    parser = argparse.ArgumentParser(description="Run ONNX model multiple times with random inputs and save execution data.")
    parser.add_argument("--model", type=str, required=True, help="your ONNX model.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of randomized inference runs.")
    args = parser.parse_args()
    
    model_name = args.model
    model_path = f"datasets/models/{model_name}/{model_name}.onnx"
    iterations = args.iterations
    
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return
    
    try:
        # Load model to verify it's valid
        model = onnx.load(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Get only runtime input shapes
    try:
        input_shapes = get_runtime_input_shapes(model_path)
        print(f"Runtime input shapes: {input_shapes}")
    except Exception as e:
        print(f"Failed to get input shapes: {e}")
        return
    
    if not input_shapes:
        print("No runtime inputs found in the model")
        return
    
    user_tests = []
    
    for i in range(iterations):
        try:
            inputs, outputs = run_onnx_inference(model_path, input_shapes)
            
            # Create test case
            # Assuming single input and single output for simplicity
            input_names = list(inputs.keys())
            output_names = list(outputs.keys())
            
            if len(input_names) > 0 and len(output_names) > 0:
                # Use the first input and first output
                in_key = input_names[0]
                out_key = output_names[0]
                
                in_array = inputs[in_key].flatten().tolist()
                out_array = outputs[out_key].flatten().tolist()
                
                test_model_info = {
                    "name": model_name,
                    "type": "exact",
                    "input": in_array,
                    "output": out_array,
                    "expected_class": 0
                }
                user_tests.append(test_model_info)
                print(f"✓ Run {i+1}/{iterations} completed")
                print(f"  Input shape: {inputs[in_key].shape}")
                print(f"  Output shape: {outputs[out_key].shape}")
            else:
                print(f"⚠️  No valid inputs/outputs found in run {i+1}")
                
        except Exception as e:
            print(f"⚠️  Error during run {i+1}: {e}")
            traceback.print_exc()
    
    if user_tests:
        # Save all test cases
        test_file_name = f"datasets/models/{model_name}/user_tests.json"
        os.makedirs(os.path.dirname(test_file_name), exist_ok=True)
        
        with open(test_file_name, 'w') as f:
            json.dump(user_tests, f, indent=2)
        
        print(f"\n✅ {len(user_tests)} tests saved to: {test_file_name}")
    else:
        print("\n⚠️  No tests were generated")

if __name__ == "__main__":
    main()