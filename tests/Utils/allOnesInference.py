#!/usr/bin/env python3
"""
ONNX Model Runner - Fills inputs with ones and runs inference
"""

import numpy as np
import onnxruntime as ort
import argparse
import sys
from pathlib import Path

def load_and_run_onnx(model_path):
    """
    Load ONNX model, create inputs filled with ones, and run inference
    
    Args:
        model_path (str): Path to the ONNX model file
    """
    
    # Check if model file exists
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    try:
        # Create inference session
        print(f"Loading ONNX model: {model_path}")
        session = ort.InferenceSession(model_path)
        
        # Get model input information
        input_info = session.get_inputs()
        output_info = session.get_outputs()
        
        print(f"\nModel has {len(input_info)} input(s) and {len(output_info)} output(s)")
        
        # Create input tensors filled with ones
        input_dict = {}
        
        for i, inp in enumerate(input_info):
            name = inp.name
            shape = inp.shape
            dtype = inp.type
            
            print(f"\nInput {i+1}:")
            print(f"  Name: {name}")
            print(f"  Shape: {shape}")
            print(f"  Type: {dtype}")
            
            # Handle dynamic shapes (replace None or -1 with 1)
            actual_shape = []
            for dim in shape:
                if dim is None or dim == -1:
                    actual_shape.append(1)
                else:
                    actual_shape.append(dim)
            
            # Convert ONNX type to numpy dtype
            if 'float' in dtype:
                np_dtype = np.float32
            elif 'double' in dtype:
                np_dtype = np.float64
            elif 'int64' in dtype:
                np_dtype = np.int64
            elif 'int32' in dtype:
                np_dtype = np.int32
            else:
                np_dtype = np.float32  # default fallback
            
            # Create tensor filled with ones
            input_tensor = np.ones(actual_shape, dtype=np_dtype)
            input_dict[name] = input_tensor
            
            print(f"  Created input tensor with shape: {input_tensor.shape}")
            print(f"  Tensor dtype: {input_tensor.dtype}")
        
        # Run inference
        print(f"\nRunning inference...")
        outputs = session.run(None, input_dict)
        
        # Display output information
        print(f"\nInference completed successfully!")
        for i, (output, out_info) in enumerate(zip(outputs, output_info)):
            print(f"\nOutput {i+1}:")
            print(f"  Name: {out_info.name}")
            print(f"  Shape: {output.shape}")
            print(f"  Type: {output.dtype}")
            print(f"  Min value: {np.min(output):.6f}")
            print(f"  Max value: {np.max(output):.6f}")
            print(f"  Mean value: {np.mean(output):.6f}")
            
            # Show a few sample values for small outputs
            if output.size <= 20:
                print(f"  Values: {output.flatten()}")
            else:
                print(f"  First 5 values: {output.flatten()[:5]}")
        
        return outputs
        
    except Exception as e:
        print(f"Error running ONNX model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Run ONNX model with inputs filled with ones"
    )
    parser.add_argument(
        "model_path", 
        help="Path to the ONNX model file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("ONNX Runtime version:", ort.__version__)
        print("Available providers:", ort.get_available_providers())
        print()
    
    # Run the model
    outputs = load_and_run_onnx(args.model_path)
    
    if outputs is not None:
        print(f"\n✓ Successfully ran model with {len(outputs)} output(s)")
    else:
        print("\n✗ Failed to run model")
        sys.exit(1)

if __name__ == "__main__":
    main()
