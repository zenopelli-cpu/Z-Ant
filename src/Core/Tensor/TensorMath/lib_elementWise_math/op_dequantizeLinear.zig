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

pub fn dequantizeLinear(
    comptime InputType: anytype,
    comptime OutputType: anytype,
    x: *const Tensor(InputType), //T1
    x_scale: *const Tensor(OutputType), //T2
    x_zero_point: ?*const Tensor(InputType), //T1
    axis: i32,
    block_size: i32,
) !Tensor(OutputType) {
    var out = try Tensor(OutputType).fromShape(x.allocator, x.shape);

    try dequantizeLinear_lean(
        InputType,
        OutputType,
        x,
        x_scale,
        x_zero_point,
        axis,
        block_size,
        &out,
    );
    return out;
}

pub inline fn dequantizeLinear_lean(
    InputType: anytype,
    comptime OutputType: anytype,
    x: *const Tensor(InputType), //T1
    x_scale: *const Tensor(OutputType), //T2=T3
    x_zero_point: ?*const Tensor(InputType), //T1
    axis: i32,
    block_size: i32,
    y: *Tensor(OutputType), //T3
) !void {
    const N = x.size;
    const rank = x.shape.len;
    const is_axis = x_scale.data.len > 1;
    const is_block = block_size > 0;

    // Normalize axis
    const normalized_axis: usize = if (axis < 0)
        @intCast(@as(i32, @intCast(rank)) + axis)
    else
        @intCast(axis);

    // Pre-compute strides for the target axis if needed
    var axis_stride: usize = 1;
    if (is_axis or is_block) {
        for (normalized_axis + 1..rank) |i| {
            axis_stride *= x.shape[i];
        }
    }

    for (0..N) |i| {
        var idx: usize = 0;
        var zp_val: i128 = 0;

        if (x_zero_point) |zp| {
            if (is_block) {
                // For blocked quantization
                const dim_size = x.shape[normalized_axis];
                const axis_coord = (i / axis_stride) % dim_size;
                const block_idx = axis_coord / @as(usize, @intCast(block_size));

                // Calculate the linear index in the scale/zero_point tensor
                // by mapping the multi-dimensional coordinate
                var linear_idx: usize = 0;
                var temp_i = i;
                var stride_scale: usize = 1;

                for (0..rank) |dim| {
                    const dim_idx = rank - 1 - dim;
                    const coord = temp_i % x.shape[dim_idx];
                    temp_i /= x.shape[dim_idx];

                    if (dim_idx == normalized_axis) {
                        // Use block index for the axis dimension
                        linear_idx += block_idx * stride_scale;
                    } else {
                        linear_idx += coord * stride_scale;
                    }

                    if (dim_idx == normalized_axis) {
                        stride_scale *= (dim_size + @as(usize, @intCast(block_size)) - 1) / @as(usize, @intCast(block_size));
                    } else {
                        stride_scale *= x.shape[dim_idx];
                    }
                }
                idx = linear_idx;
                zp_val = @as(i128, zp.data[idx]);
            } else if (is_axis) {
                // For per-axis quantization
                const axis_coord = (i / axis_stride) % x.shape[normalized_axis];
                idx = axis_coord;
                zp_val = @as(i128, zp.data[idx]);
            } else {
                // Per-tensor quantization
                idx = 0;
                zp_val = @as(i128, zp.data[0]);
            }
        } else {
            // No zero_point specified → default 0
            zp_val = 0;
            idx = 0;
        }

        const scale: OutputType = x_scale.data[idx];
        const xval: InputType = x.data[i];
        const deq = @as(OutputType, @floatFromInt(@as(i128, xval) - zp_val)) * scale;
        y.data[i] = deq;
    }
}
