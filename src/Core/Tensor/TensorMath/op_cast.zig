const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const DataType = zant.onnx.DataType; // Assuming DataType enum is accessible

/// Casts elements of an input tensor to a specified data type.
/// This is a 'lean' version that writes to a pre-allocated output tensor.
/// INPUT input:  Input tensor of type T1.
/// INPUT to:     Target DataType enum value.
/// OUTPUT output: Output tensor of type T2 (determined by 'to').
pub fn cast_lean(
    comptime T1: type, // Input type
    comptime T2: type, // Output type (must match 'to' argument's implied type)
    input: *const Tensor(T1),
    output: *Tensor(T2),
    to_dtype: DataType,
) !void {
    // --- Basic Validations ---
    if (!std.meta.eql(T2, zant.onnx.dataTypeToZigType(to_dtype))) {
        // This should ideally be caught during code generation, but good for safety
        return TensorMathError.InvalidDataType;
    }
    if (!std.mem.eql(usize, input.shape, output.shape)) {
        return TensorMathError.ShapeMismatch;
    }
    if (input.size != output.size) {
        // Should not happen if shapes match, but safety check
        return TensorMathError.InvalidDimensions;
    }

    // --- Casting Logic ---
    // TODO: Add comprehensive casting rules as per ONNX spec when more types (float8, string) are supported.
    // For now, implementing direct numeric/bool casts.

    var i: usize = 0;
    while (i < input.size) : (i += 1) {
        const input_val = input.data[i];
        output.data[i] = switch (T1) {
            f32 => switch (T2) {
                f32 => input_val, // No change
                i64 => @intFromFloat(input_val),
                i32 => @intFromFloat(input_val), // Assuming i32 might be needed
                i8 => @intFromFloat(input_val),
                u8 => @intFromFloat(input_val),
                bool => input_val != 0.0,
                else => return TensorMathError.InvalidDataType, // Unsupported target type
            },
            i64 => switch (T2) {
                f32 => @floatFromInt(input_val),
                i64 => input_val, // No change
                i32 => @intCast(input_val), // Truncates
                i8 => @intCast(input_val), // Truncates
                u8 => @intCast(input_val), // Truncates
                bool => input_val != 0,
                else => return TensorMathError.InvalidDataType,
            },
            i32 => switch (T2) { // Example if i32 is used
                f32 => @floatFromInt(input_val),
                i64 => @intCast(input_val),
                i32 => input_val,
                i8 => @intCast(input_val),
                u8 => @intCast(input_val),
                bool => input_val != 0,
                else => return TensorMathError.InvalidDataType,
            },
            i8 => switch (T2) {
                f32 => @floatFromInt(input_val),
                i64 => @intCast(input_val),
                i32 => @intCast(input_val),
                i8 => input_val,
                u8 => @intCast(input_val), // Reinterprets if negative
                bool => input_val != 0,
                else => return TensorMathError.InvalidDataType,
            },
            u8 => switch (T2) {
                f32 => @floatFromInt(input_val),
                i64 => @intCast(input_val),
                i32 => @intCast(input_val),
                i8 => @intCast(input_val), // Reinterprets if > 127
                u8 => input_val,
                bool => input_val != 0,
                else => return TensorMathError.InvalidDataType,
            },
            bool => switch (T2) {
                // Assuming T is the actual Zig type corresponding to the DataType
                // f32 => @boolToInt(input_val),
                // i64 => @boolToInt(input_val),
                // i32 => @boolToInt(input_val),
                // i8 => @boolToInt(input_val),
                // u8 => @boolToInt(input_val),
                bool => input_val,
                else => return TensorMathError.InvalidDataType,
            },
            else => return TensorMathError.InvalidDataType, // Unsupported source type
        };
    }
}
