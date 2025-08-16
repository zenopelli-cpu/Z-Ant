const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

// Followint the onnx standard
// https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
//Saturation is done according to:
// uint16: [0, 65535]
// int16: [-32768, 32767]
// uint8: [0, 255]
// int8: [-128, 127]
// uint4: [0, 15]
// int4: [-8, 7]

pub fn quantizeLinear(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    x: *Tensor(InputType), //T1
    y_scale: *Tensor(InputType), //T2
    y_zero_point: ?*Tensor(OutputType), //T3
    axis: i32,
    block_size: i32,
    // output_dtype: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // precision: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // saturate: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
) !Tensor(OutputType) {

    // Create output tensor shape copy
    var out = try Tensor(OutputType).fromShape(x.allocator, x.shape);

    // Dispatch to lean function
    try quantizeLinear_lean(
        InputType,
        OutputType,
        x,
        y_scale,
        y_zero_point,
        axis,
        block_size,
        // output_dtype,
        // precision,
        // saturate,
        &out,
    );
    return out;
}

pub inline fn quantizeLinear_lean(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    x: *Tensor(InputType), //T1
    y_scale: *Tensor(InputType), //T2
    y_zero_point: ?*Tensor(OutputType), //T3=T4
    axis: i32,
    block_size: i32,
    // output_dtype: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // precision: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    // saturate: i32, only used when parsing, see IR_zant/op_union_operators/op_quantizeLinear
    y: *Tensor(OutputType), //T4
) !void {

    // quantization formula: y = saturate((x / y_scale) + y_zero_point)
    // three supported quantization granularities.
    //      Per-tensor (per-layer) quantization: y_scale is a scalar.
    //      Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape (D0, ..., Di, ..., Dn) and axis=i, y_scale is a 1-D tensor of length Di.
    //      Blocked quantization: The scale’s shape is identical to the input’s shape, except for one dimension, in which blocking is performed. Given x shape (D0, ..., Di, ..., Dn), axis=i, and block size B: y_scale shape is (D0, ..., ceil(Di/B), ..., Dn).

    const N = x.size;
    const rank = @as(i32, @intCast(x.shape.len));

    // if (y_zero_point.shape.len != y_scale.shape.len) return error.Invalid_zeroPoint_or_scale;
    // for (0..y_zero_point.shape.len) |i| {
    //     if (y_zero_point.data[i] != y_scale.data[i]) return error.Invalid_zeroPoint_or_scale;
    // }

    const is_perTensor: bool = y_scale.shape.len == 1 and y_scale.data.len == 1;
    const is_perAxis: bool = !is_perTensor and y_scale.data.len == x.shape[@as(usize, @intCast(axis))] and y_scale.shape.len == 1;
    const is_perBlock: bool = false;

    //check if is_perBlock
    if (!is_perTensor and !is_perAxis) {
        if (y_scale.shape.len != x.shape.len) return error.Invalid_perBlock_scale; // The scale’s shape is identical to the input’s shape -> same lenght

        for (0..y_scale.shape.len) |i| { //The scale’s shape is identical to the input’s shape
            if (i == axis) { //dimension where bloking is performed
                if (block_size <= 0) return error.Invalid_block_size;
                if (y_scale.shape[i] != @divFloor(@as(i32, @intCast(x.shape[i])), block_size)) return error.Invalid_scale_shape;
            } else {
                if (y_scale.shape[i] != x.shape[i]) return error.Invalid_scale_shape;
            }
        }
    }

    // DEBUG
    // std.debug.print("\nx: ", .{});
    // print(InputType, x);
    // std.debug.print("\ny_scale: ", .{});
    // print(InputType, y_scale);
    // std.debug.print("\ny_zero_point: ", .{});
    // if (y_zero_point) |zp| print(OutputType, zp) else std.debug.print("null ", .{});
    // std.debug.print("\naxis: {}", .{axis});
    // std.debug.print("\nblock_size: {}", .{block_size});
    // std.debug.print("\n: -->{s}", .{if (is_perBlock) "is_block" else if (is_perAxis) "is_axis" else "is_tensor"});

    if (is_perBlock) {
        // Blocked quantization: different scale/zero_point for each block
        const axis_u = @as(usize, @intCast(axis));
        const block_size_u = @as(usize, @intCast(block_size));

        // Calculate strides for multi-dimensional indexing
        var x_strides: [4]usize = undefined; // assuming max 4 dimensions
        var scale_strides: [4]usize = undefined;

        // Calculate strides for x tensor
        x_strides[rank - 1] = 1;
        for (0..rank - 1) |i| {
            const idx = rank - 2 - i;
            x_strides[idx] = x_strides[idx + 1] * x.shape[idx + 1];
        }

        // Calculate strides for scale tensor
        scale_strides[rank - 1] = 1;
        for (0..rank - 1) |i| {
            const idx = rank - 2 - i;
            scale_strides[idx] = scale_strides[idx + 1] * y_scale.shape[idx + 1];
        }

        // Iterate through all elements
        for (0..N) |i| {

            // Calculate multi-dimensional indices for x
            var x_indices: [8]usize = undefined;
            var remaining = i;
            for (0..rank) |dim| {
                x_indices[dim] = remaining / x_strides[dim];
                remaining %= x_strides[dim];
            }

            // Calculate corresponding scale indices (block index for axis dimension)
            var scale_indices: [8]usize = undefined;
            for (0..rank) |dim| {
                if (dim == axis_u) {
                    scale_indices[dim] = x_indices[dim] / block_size_u;
                } else {
                    scale_indices[dim] = x_indices[dim];
                }
            }

            // Calculate flat index for scale
            var scale_idx: usize = 0;
            for (0..rank) |dim| {
                scale_idx += scale_indices[dim] * scale_strides[dim];
            }

            // Apply quantization formula
            const scale_val = y_scale.data[scale_idx];
            const zero_point_val = if (y_zero_point) |zp| zp.data[scale_idx] else 0;

            y.data[i] = try quantize(InputType, OutputType, x.data[i], scale_val, zero_point_val);
        }
    } else if (is_perAxis) {
        // Per-axis quantization: different scale/zero_point for each slice along the axis
        const axis_u = @as(usize, @intCast(axis));
        const axis_size = x.shape[axis_u];

        // Calculate stride for the axis dimension
        var axis_stride: usize = 1;
        for (axis_u + 1..x.shape.len) |i| {
            axis_stride *= x.shape[i];
        }

        // Calculate outer stride (elements between different axis slices)
        // var outer_stride: usize = axis_stride * axis_size;

        for (0..N) |i| {
            // Calculate which axis index this element belongs to
            const axis_idx = (i / axis_stride) % axis_size;

            // Apply quantization formula
            const scale_val = y_scale.data[axis_idx];
            const zero_point_val = if (y_zero_point) |zp| zp.data[axis_idx] else 0;

            y.data[i] = try quantize(InputType, OutputType, x.data[i], scale_val, zero_point_val);
        }
    } else if (is_perTensor) {
        // Per-tensor quantization: same scale/zero_point for all elements
        const scale_val = y_scale.data[0];
        const zero_point_val = if (y_zero_point) |zp| zp.data[0] else 0;

        for (0..N) |i| {
            y.data[i] = try quantize(InputType, OutputType, x.data[i], scale_val, zero_point_val);
        }
    }
}

fn print(comptime T: anytype, tens: *Tensor(T)) void {
    std.debug.print("\n {{", .{});
    for (0..tens.size) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{}", .{tens.data[i]});
    }
    std.debug.print(" }}", .{});
}

inline fn quantize(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    inputData: InputType,
    scale: InputType,
    zp: OutputType,
) !OutputType {
    const scaled: InputType = inputData / scale;
    // OSS: For (x / y_scale), it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
    var rounded = @round(scaled); // oss: @TypeOf(rounded) == TypeOf(scale) == InputType
    rounded = if (scaled - rounded == 0.5 and @rem(rounded, 2) == 1)
        rounded - 1
    else
        rounded;

    const quantized = rounded + @as(InputType, @floatFromInt(zp));

    return saturate(InputType, OutputType, quantized);
}

// Helper function to saturate values according to output type
inline fn saturate(comptime InputType: type, comptime OutputType: type, value: InputType) OutputType {
    const info = @typeInfo(OutputType);
    if (info == .int) {
        const min_val: InputType = std.math.minInt(OutputType);
        const max_val: InputType = std.math.maxInt(OutputType);
        return @as(OutputType, @intFromFloat(std.math.clamp(value, min_val, max_val)));
    } else {
        // For float types, just convert directly
        return @as(OutputType, value);
    }
}
