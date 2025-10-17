# ONNX Operator Template – Zant SDK

This template provides the structure and guidance for implementing a new ONNX operator.

---

## ✅ What You Should Implement

Each ONNX operator should be implemented as a `struct` that provides the following methods:

### Required Methods

- `init(nodeProto: *NodeProto) !YourOpStruct`  
  Parse input tensors and ONNX attributes. This method builds the operator instance.

- `pub fn get_output_shape(self: YourOpStruct) []usize `
  Returns the shape of the output

- ` pub fn get_input_tensors(self: YourOpStruct) ![]*TensorZant`
  Returns all the input tensors

- ` pub fn get_output_tensors(self: YourOpStruct) ![]*TensorZant`
  Returns all the output tensors
- `write_op(self: YourOpStruct, writer: *std.Io.Writer) !void`  
  Emits backend-compatible code that performs the actual tensor operation.

- `compute_output_shape(self: YourOpStruct) []usize`  
  Computes the output tensor shape (may use broadcasting utilities).

- `pub fn print(self: YourOpStruct) void`
  Debug function

---

## 🧠 Attribute Parsing Guide

ONNX attributes can be of various types. Here’s how they map to Zig:

| ONNX Type | Zig Type          | Notes                              |
| --------- | ----------------- | ---------------------------------- |
| INT       | `?i64`            | Optional integer attribute         |
| FLOAT     | `?f32`            | Optional floating-point attribute  |
| STRING    | `?[]const u8`     | Optional string (e.g., mode, axes) |
| BOOL      | `?bool` or as INT | ONNX sometimes uses INT for bool   |

Optional input tensors can be checked using:

## Important :

When implementing a mathematical operator be as much coherent as possible to the standard onnx documentation: [onnx operators](https://onnx.ai/onnx/operators/)
