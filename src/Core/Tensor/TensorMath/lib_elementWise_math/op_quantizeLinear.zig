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
    x: *const Tensor(InputType), //T1
    y_scale: *const Tensor(InputType), //T2
    y_zero_point: ?*const Tensor(OutputType), //T3
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
    const rank = x.shape.len;
    const is_axis = y_scale.data.len > 1;
    const is_block = block_size > 0;

    for (0..N) |i| {
        // Determine index into scale/zero_point:
        var idx: usize = 0;
        var zp: i128 = 0;
        if (is_block) {
            const dim = if (axis < 0) rank + axis else axis;
            const dim_size = x.shape[dim];
            const block_count = (dim_size + block_size - 1) / block_size;
            const stride = block_size;
            const pos = (i / stride) % block_count;
            idx = pos;
            zp = @as(i128, y_zero_point.?.data[idx]);
        } else if (is_axis) {
            const dim = if (axis < 0) rank + axis else axis;
            const dim_size = x.shape[dim];
            const stride_per = x.size / dim_size;
            idx = (i / stride_per) % dim_size;
            zp = @as(i128, y_zero_point.?.data[idx]);
        } else {
            idx = 0;
        }

        const s: InputType = y_scale.data[idx];
        const xf: InputType = @as(InputType, x.data[i]);
        const scaled = xf / s;
        const rounded = @round(scaled);
        rounded = if (scaled - rounded == 0.5 and rounded % 2 == 1) (if (scaled > 0) rounded - 1 else rounded - 1);
        const q = rounded + zp;

        //saturate
        const clamped: i128 = switch (OutputType) {
            u16 => std.math.clamp(q, 0, 65535),
            i16 => std.math.clamp(q, -32768, 32767),
            u8 => std.math.clamp(q, 0, 255),
            i8 => std.math.clamp(q, -128, 127),
            // For uint4/int4, match as exact type aliases (e.g. u4Type, i4Type)
            u4 => std.math.clamp(q, 0, 15),
            i4 => std.math.clamp(q, -8, 7),
            else => q,
        };

        y.data[i] = @as(OutputType, clamped);
    }
}
