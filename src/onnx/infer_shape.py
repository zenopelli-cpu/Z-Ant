import argparse
import onnx
from onnx import shape_inference
from onnxsim import simplify


def main():
    print(f"\n __main__")
    parser = argparse.ArgumentParser(description=" Upgrade your model with all intermediate tensor's shapes")
    parser.add_argument("--path", type=str, required=True, help="Path of your model.")
    args = parser.parse_args()
    
    model = onnx.load(args.path)

    inferred_model = shape_inference.infer_shapes(model)

    model_simp, check = simplify(inferred_model)

    if check:
        print("Simplified model is valid!")
        onnx.save(model_simp, args.path)
    else:
        raise RuntimeError("Something went wrong in the onnx simplifier()")

    
if __name__ == "__main__":
    main()