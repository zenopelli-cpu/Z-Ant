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

pub fn dequantizeLinear(comptime InputType: anytype, comptime OutputType: anytype, x: *const Tensor(InputType), y_scale: *const Tensor(OutputType), y_zero_point: ?*const Tensor(InputType), axis: i32, block_size: i32) !Tensor(OutputType) {
    var out = try Tensor(OutputType).fromShape(x.allocator, x.shape);

    try dequantizeLinear_lean(
        InputType,
        OutputType,
        x,
        y_scale,
        y_zero_point,
        axis,
        block_size,
        &out,
    );
    return out;
}

pub inline fn dequantizeLinear_lean(
    InputType: anytype,
    comptime OutputType: anytype,
    x: *const Tensor(InputType),
    y_scale: *const Tensor(OutputType),
    y_zero_point: ?*const Tensor(InputType),
    axis: i32,
    block_size: i32,
    y: *Tensor(OutputType),
) !void {
    const N = x.size;
    const rank = x.shape.len;
    const is_axis = y_scale.data.len > 1;
    const is_block = block_size > 0;

    for (0..N) |i| {
        var idx: usize = 0;
        var zp_val: i128 = 0;

        if (y_zero_point) |zp| {
            if (is_block) {
                const dim = if (axis < 0) rank + axis else axis;
                const dim_size = x.shape[dim];
                const block_count = (dim_size + block_size - 1) / block_size;
                const stride = block_size;
                const pos = (i / stride) % block_count;
                idx = pos;
                zp_val = @as(i128, zp.data[idx]);
            } else if (is_axis) {
                const dim = if (axis < 0) rank + axis else axis;
                const dim_size = x.shape[dim];
                const stride_per = x.size / dim_size;
                idx = (i / stride_per) % dim_size;
                zp_val = @as(i128, zp.data[idx]);
            } else {
                idx = 0;
                zp_val = @as(i128, zp.data[0]);
            }
        } else {
            // y_zero_point not specified → default 0
            zp_val = 0;
            idx = 0;
        }

        const scale: OutputType = y_scale.data[idx];
        const xval: InputType = x.data[i];
        const deq = @as(OutputType, @floatFromInt(@as(i128, xval) - zp_val)) * scale;

        y.data[i] = deq;
    }
}
