import onnx
import onnxruntime as ort
import argparse
import json
import numpy as np
import os
import traceback

def get_input_shapes(model):
    input_shapes = {}
    for input_tensor in model.graph.input:
        name = input_tensor.name
        dims = [d.dim_value if (d.dim_value > 0) else 1 for d in input_tensor.type.tensor_type.shape.dim]
        input_shapes[name] = dims
    return input_shapes

def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32)

def run_onnx_inference(model_path, input_shapes):
    sess = ort.InferenceSession(model_path)
    inputs = {name: generate_random_input(shape) for name, shape in input_shapes.items()}
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
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    input_shapes = get_input_shapes(model)
    print(f"Model input shapes: {input_shapes}")

    user_tests = []

    for i in range(iterations):
        try:
            inputs, outputs = run_onnx_inference(model_path, input_shapes)

            for in_key, out_key in zip(inputs.keys(), outputs.keys()):
                in_array = inputs[in_key].flatten().tolist()
                out_array = outputs[out_key].flatten().tolist()

                test_model_info = {
                    "name": os.path.basename(model_path),
                    "type": "exact",
                    "input": in_array,
                    "output": out_array,
                    "expected_class": 0
                }
                user_tests.append(test_model_info)

            print(f"✓ Run {i+1}/{iterations} completed")
        except Exception as e:
            print(f"⚠️  Error during run {i+1}: {e}")
            traceback.print_exc()

    # Save all test cases
    test_file_name = "generated/"+model_name + "/user_tests.json"
    with open(test_file_name, 'w') as f:
        json.dump(user_tests, f, indent=2)

    print(f"\n✅ All tests saved to: {test_file_name}")

if __name__ == "__main__":
    main()
