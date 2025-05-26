# ONNX Operator Template – Zant SDK

This template provides the structure and guidance for implementing a new ONNX operator.

---

## ✅ What You Should Implement

Each ONNX operator should be implemented as a `struct` that provides the following methods:

### Required Methods

- `init(nodeProto: *NodeProto) !YourOpStruct`  
  Parse input tensors and ONNX attributes. This method builds the operator instance.

- `compute_output_shape(self: YourOpStruct) []usize`  
  Computes the output tensor shape (may use broadcasting utilities).

- `write_op(self: YourOpStruct, writer: std.fs.File.Writer) !void`  
  Emits backend-compatible code that performs the actual tensor operation.

### Optional Methods

- `get_output_shape(self: YourOpStruct) []usize`  
  Returns output tensor shape for external use.

- `get_output_tensor(self: YourOpStruct) *TensorZant`  
  Returns a pointer to the output tensor object.

- `print(self: YourOpStruct)`  
  Debugging helper to display internal state.

---

## 🧠 Attribute Parsing Guide

ONNX attributes can be of various types. Here’s how they map to Zig:

| ONNX Type | Zig Type     | Notes                                  |
|-----------|--------------|----------------------------------------|
| INT       | `?i64`       | Optional integer attribute             |
| FLOAT     | `?f32`       | Optional floating-point attribute      |
| STRING    | `?[]const u8`| Optional string (e.g., mode, axes)     |
| BOOL      | `?bool` or as INT | ONNX sometimes uses INT for bool |

Optional input tensors can be checked using:

```C++
if (nodeProto.input.len > 1 and nodeProto.input[1].len > 0) { ... }

---



const std = @import("std");
const allocator = std.heap.page_allocator;

const zant = @import("zant");
const tensorMath = zant.core.tensor.math_standard;

// --- ONNX ---
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

// --- Zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const TensorCategory = tensorZant.TensorCategory;

const utils = @import("codegen").utils;

// https://onnx.ai/onnx/operators/onnx__<OperatorName>.html
// INPUTS:
//      - <input_name> (<optional>) - T<data_type>: Description of the input.
//      - ... (one line per input)
// OUTPUTS:
//      - <output_name> (<optional>) - T<data_type>: Description of the output.
//      - ...
// ATTRIBUTES:
//      - <attribute_name> (<default>) (<optional>) - type: Description of the attribute, including default values if any.
//      - ...

pub const NewOp = struct {
    // --- Inputs ---
    input_X: *TensorZant,           // Required input tensor
    input_Y: ?*TensorZant = null,   // Optional input tensor (can be null)

    // --- Outputs ---
    output_Z: *TensorZant,

    // --- Attributes ---
    attr_int: ?i64 = null,           // Example: number of elements, axis, or similar
    attr_float: ?f32 = null,         // Example: scaling factor
    attr_str: ?[]const u8 = null,    // Example: mode ("linear", "nearest", etc.)
    attr_bool: bool = false,         // Example: a flag (e.g., keepdims)

    /// Initialize the operator from an ONNX NodeProto
    pub fn init(nodeProto: *NodeProto) !NewOp {
        // --- Required input ---
        const input_X = tensorZant.tensorMap.getPtr(nodeProto.input[0]) orelse return error.input_X_notFound;

        // --- Optional second input ---
        const input_Y = if (nodeProto.input.len > 1)
            tensorZant.tensorMap.getPtr(nodeProto.input[1]) orelse return error.input_Y_notFound
            else
            null;

        // --- Output tensor ---
        const output_Z = tensorZant.tensorMap.getPtr(nodeProto.output[0]) orelse return error.output_Z_notFound;

        // --- Optional second output ---
        const input_Y = if (nodeProto.output.len > 1)
            tensorZant.tensorMap.getPtr(nodeProto.input[1]) orelse return error.input_Y_notFound
            else
            null;

        // --- Attribute parsing ---
        var attr_int: ?i64 = null;
        var attr_float: ?f32 = null;
        var attr_str: ?[]const u8 = null;
        var attr_bool: bool = false;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                // INT type attribute: commonly used for axes, counts, etc.
                if (attr.type == onnx.AttributeType.INT) attr_int = attr.i;
            } else if (std.mem.eql(u8, attr.name, "scale")) {
                // FLOAT type attribute: useful for scaling/normalization
                if (attr.type == onnx.AttributeType.FLOAT) attr_float = attr.f32;
            } else if (std.mem.eql(u8, attr.name, "mode")) {
                // STRING type attribute: often used for modes (e.g., "nearest")
                if (attr.type == onnx.AttributeType.STRING) attr_str = attr.s;
            } else if (std.mem.eql(u8, attr.name, "keepdims")) {
                // BOOL-like attribute: encoded as INT (0 = false, 1 = true)
                if (attr.type == onnx.AttributeType.INT) attr_bool = attr.i != 0;
            }
        }

        return NewOp{
            .input_X = input_X,
            .input_Y = input_Y,
            .output_Z = output_Z,
            .attr_int = attr_int,
            .attr_float = attr_float,
            .attr_str = attr_str,
            .attr_bool = attr_bool,
        };
    }

    /// Compute the output shape, possibly using broadcasting rules
    pub fn compute_output_shape(self: NewOp) []usize {
        var output_shape: []usize = undefined;

        if (self.input_Y) |input_y| {
            output_shape = utils.broadcastShapes(allocator, self.input_X.shape, input_y.shape) catch unreachable;
        } else {
            output_shape = allocator.dupe(usize, self.input_X.shape) catch unreachable;
        }

        self.output_Z.shape = output_shape;
        return output_shape;
    }

    pub fn get_output_shape(self: NewOp) []usize {
        return self.output_Z.getShape();
    }

    pub fn get_output_tensor(self: NewOp) *TensorZant {
        return self.output_Z;
    }

    /// Write the tensor operation to the generated output file
    pub fn write_op(self: NewOp, writer: std.fs.File.Writer) !void {
       //input tensors 

        //required tensor
        var input_X_string: []u8 = undefined;
        defer allocator.free(input_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            input_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            input_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_A.name),
            });
        }

        //optional tensor
        var tensor_split_string: []u8 = undefined;
        defer allocator.free(tensor_split_string);

        if (self.input_Y != null) {
            //AS ABOVE
        }

        //NOTE: str attributes may need transformation like:
        var attr_str: []const u8 = undefined;
        attr_str = try utils.i64SliceToUsizeArrayString(self.attr_str);
        // other useful transformation can be found in: 
        // src\CodeGen\utils.zig

         _ = try writer.print(
            \\    
            \\
            \\    tensMath.NewOp_lean(
            \\        T
            \\        {s},
            \\        {s},
            \\        {d}, 
            \\        {d}, 
            \\        {s},
            \\        &tensor_{s}, 
            \\    )
        ,.{
            //inputs
            input_X_string,
            input_Y_string, 
            //attributes
            self.attr, 
            self.attr_float,
            attr_str,
            self.attr_bool,
            //outputs
            try utils.getSanitizedName(self.output.name), // output
        });
    }

    pub fn print(self: NewOp) void {
        std.debug.print("\n NewOp Debug Info:\n  input_X: {}\n  input_Y: {}\n  output_Z: {}\n  int: {}\n  float: {}\n  str: {?s}\n  bool: {}\n",
            .{ self.input_X.name, self.input_Y orelse null, self.output_Z.name, self.attr_int, self.attr_float, self.attr_str, self.attr_bool });
    }
};
