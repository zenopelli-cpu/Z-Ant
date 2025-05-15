const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("../../../zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const TensorCategory = tensorZant.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = @import("../../../CodeGen/utils.zig");

// https://onnx.ai/onnx/operators/onnx__Constant.html
// Outputs:
// - output (heterogeneous) - T: Output tensor containing the same value of the provided tensor.
// Attributes - only one of these should be specified:
// - value (TENSOR): The value for the elements of the output tensor.
// - sparse_value (SPARSE_TENSOR): The value for the elements of the output tensor in sparse format.
// - value_float (FLOAT): The value for the sole element for the scalar, float32, output tensor.
// - value_floats (FLOATS): The values for the elements for the 1D, float32, output tensor.
// - value_int (INT): The value for the sole element for the scalar, int64, output tensor.
// - value_ints (INTS): The values for the elements for the 1D, int64, output tensor.
// - value_string (STRING): The value for the sole element for the scalar, UTF-8 string, output tensor.
// - value_strings (STRINGS): The values for the elements for the 1D, UTF-8 string, output tensor.

pub const Constant = struct {
    output: *TensorZant,
    // attributes:
    value: ?*TensorZant,
    sparse_value: ?*TensorZant,
    value_float: ?f32,
    value_floats: ?[]f32,
    value_int: ?i64,
    value_ints: ?[]i64,
    value_string: ?[]const u8,
    value_strings: ?[][]const u8,

    pub fn init(nodeProto: *NodeProto) !Constant {
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;
        var value: ?*TensorZant = null;
        var sparse_value: ?*TensorZant = null;
        var value_float: ?f32 = null;
        var value_floats: ?[]f32 = null;
        var value_int: ?i64 = null;
        var value_ints: ?[]i64 = null;
        var value_string: ?[]const u8 = null;
        var value_strings: ?[][]const u8 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "value")) |_| {
                if (attr.type == onnx.AttributeType.TENSOR) value = attr.t;
            } else if (std.mem.indexOf(u8, attr.name, "sparse_value")) |_| {
                if (attr.type == onnx.AttributeType.SPARSE_TENSOR) sparse_value = attr.sparse_tensor;
            } else if (std.mem.indexOf(u8, attr.name, "value_float")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) value_float = attr.f;
            } else if (std.mem.indexOf(u8, attr.name, "value_floats")) |_| {
                if (attr.type == onnx.AttributeType.FLOATS) value_floats = attr.floats;
            } else if (std.mem.indexOf(u8, attr.name, "value_int")) |_| {
                if (attr.type == onnx.AttributeType.INT) value_int = attr.i;
            } else if (std.mem.indexOf(u8, attr.name, "value_ints")) |_| {
                if (attr.type == onnx.AttributeType.INTS) value_ints = attr.ints;
            } else if (std.mem.indexOf(u8, attr.name, "value_string")) |_| {
                if (attr.type == onnx.AttributeType.STRING) value_string = attr.s;
            } else if (std.mem.indexOf(u8, attr.name, "value_strings")) |_| {
                if (attr.type == onnx.AttributeType.STRINGS) value_strings = attr.strings;
            }
        }

        return Constant{
            .output = output,
            .value = value,
            .sparse_value = sparse_value,
            .value_float = value_float,
            .value_floats = value_floats,
            .value_int = value_int,
            .value_ints = value_ints,
            .value_string = value_string,
            .value_strings = value_strings,
        };
    }
};
