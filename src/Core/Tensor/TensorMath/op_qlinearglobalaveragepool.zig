const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensorError = zant.utils.error_handler.TensorError;

const accelerators = @import("../Accelerators/mod.zig");
const cmsis_dsp = @import("../Accelerators/stm32n6/cmsis_dsp.zig");
// cmsis_nn avgpool expects NHWC; our tensors are NCHW. We avoid it here.

const tensor_module = zant.core.tensor;

pub var log_functionC: ?*const fn ([*c]u8) callconv(.C) void = null;

pub export fn setQLinearGlobalAveragePoolLogFunctionC(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_functionC = func;
}

inline fn logDebug(comptime fmt: []const u8, args: anytype) void {
    var log_ptr: ?*const fn ([*c]u8) callconv(.C) void = log_functionC;
    if (log_ptr == null) {
        log_ptr = tensor_module.log_function;
    }
    if (log_ptr) |log| {
        var buffer: [256:0]u8 = undefined;
        const msg = std.fmt.bufPrintZ(&buffer, fmt, args) catch return;
        log(@constCast(msg.ptr));
    }
}

// Import existing global average pool operation for shape calculation
const globalAvgPool = @import("op_globalAveragePool.zig");

/// QLinearGlobalAveragePool operation following ONNX specification
/// Performs quantized global average pooling using linear quantization scheme
///
/// INPUTS:
/// - x: quantized input tensor (typically int8/uint8) of shape (N, C, H, W, ...)
/// - x_scale: scale factor for input quantization
/// - x_zero_point: zero point for input quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
///
/// OUTPUT:
/// - y: quantized output tensor of shape (N, C, 1, 1, ...)
///
/// Formula: quantized_output = quantize(global_average_pool(dequantize(x)), y_scale, y_zero_point)
pub fn qlinearglobalaveragepool(
    comptime InputType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
) !Tensor(InputType) {
    // Input validation
    if (x.shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }
    if (x_scale.size != 1 or y_scale.size != 1) {
        return TensorMathError.InvalidDimensions;
    }
    if (x_zero_point.size != 1 or y_zero_point.size != 1) {
        return TensorMathError.InvalidDimensions;
    }

    // Calculate output shape (same as regular GlobalAveragePool)
    const output_shape = try get_qlinearglobalaveragepool_output_shape(x.shape);
    defer pkg_allocator.free(output_shape);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    // Perform quantized global average pooling
    try lean_qlinearglobalaveragepool(
        InputType,
        ScaleType,
        ZeroPointType,
        x,
        x_scale,
        x_zero_point,
        y_scale,
        y_zero_point,
        &output,
    );

    return output;
}

/// Lean version of QLinearGlobalAveragePool that operates on pre-allocated output tensor
pub fn lean_qlinearglobalaveragepool(
    x: anytype,
    x_scale: anytype,
    x_zero_point: anytype,
    output: anytype,
    y_scale: anytype,
    y_zero_point: anytype,
) !void {
    const x_scale_val = x_scale.data[0];
    const x_zero_point_val = x_zero_point.data[0];
    const y_scale_val = y_scale.data[0];
    const y_zero_point_val = y_zero_point.data[0];

    // Handle different tensor dimensions
    const batch_size = if (x.shape.len >= 1) x.shape[0] else 1;
    const channels = if (x.shape.len >= 2) x.shape[1] else 1;

    // Calculate spatial size based on tensor dimensions
    var spatial_size: usize = 1;
    if (x.shape.len >= 3) {
        // For 4D tensors (NCHW format): [N, C, H, W] - spatial dims are H, W
        // For 3D tensors (CHW format): [C, H, W] - spatial dims are H, W
        for (2..x.shape.len) |i| {
            spatial_size *= x.shape[i];
        }
    } else if (x.shape.len == 2) {
        // For 2D tensors: treat as [N, C] - no spatial dimensions
        spatial_size = 1;
    } else if (x.shape.len == 1) {
        // For 1D tensors: treat as [C] - no batch or spatial dimensions
        spatial_size = 1;
    }

    if (x.data.len > 0) {
        const InputType = @TypeOf(x.data[0]);
        const x_scale_f32 = std.math.lossyCast(f32, x_scale_val);
        const y_scale_f32 = std.math.lossyCast(f32, y_scale_val);
        const x_zero_point_i32 = std.math.lossyCast(i32, x_zero_point_val);
        const y_zero_point_i32 = std.math.lossyCast(i32, y_zero_point_val);

        if (tryAcceleratedQLinearGlobalAveragePool(
            InputType,
            x,
            x_scale_f32,
            x_zero_point_i32,
            output,
            y_scale_f32,
            y_zero_point_i32,
            batch_size,
            channels,
            spatial_size,
        )) {
            logDebug(
                "[ZANT][qlinearglobalavgpool] accelerated CMSIS path used: n={d} c={d} spatial={d} x_scale={d:.6} y_scale={d:.6}\n",
                .{ batch_size, channels, spatial_size, x_scale_f32, y_scale_f32 },
            );
            return;
        }
    }

    logDebug("[ZANT][qlinearglobalavgpool] using reference path\n", .{});

    // Process each batch and channel
    for (0..batch_size) |n| {
        for (0..channels) |c| {
            var sum: f64 = 0.0; // Use f64 for better precision in accumulation

            // Sum all spatial elements for this channel
            // Different indexing based on tensor dimensions
            if (x.shape.len == 1) {
                // For 1D tensors: just use the single element
                const input_idx = 0;
                const InputType = @TypeOf(x.data[0]);
                const dequant_val = if (@typeInfo(InputType) == .int)
                    (@as(f32, @floatFromInt(x.data[input_idx])) - @as(f32, @floatFromInt(x_zero_point_val))) * x_scale_val
                else
                    (x.data[input_idx] - @as(InputType, @floatFromInt(x_zero_point_val))) * x_scale_val;
                sum += @as(f64, dequant_val);
            } else {
                // For multi-dimensional tensors
                const channel_start = ((n * channels) + c) * spatial_size;
                for (0..spatial_size) |i| {
                    const input_idx = channel_start + i;
                    // Dequantize each input value and accumulate
                    const InputType = @TypeOf(x.data[0]);
                    const dequant_val = if (@typeInfo(InputType) == .int)
                        (@as(f32, @floatFromInt(x.data[input_idx])) - @as(f32, @floatFromInt(x_zero_point_val))) * x_scale_val
                    else
                        (x.data[input_idx] - @as(InputType, @floatFromInt(x_zero_point_val))) * x_scale_val;
                    sum += @as(f64, dequant_val);
                }
            }

            // Calculate average
            const spatial_size_f32 = @as(f32, @floatFromInt(spatial_size));
            const sum_f32 = @as(f32, @floatCast(sum));
            const avg_float = sum_f32 / spatial_size_f32;

            // Quantize result
            const scaled_result = (avg_float / y_scale_val) + @as(f32, @floatFromInt(y_zero_point_val));

            // Clamp to valid range for output type
            const OutputType = @TypeOf(output.data[0]);
            const min_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.minInt(OutputType)))
            else
                std.math.floatMin(OutputType);
            const max_val = if (@typeInfo(OutputType) == .int)
                @as(f32, @floatFromInt(std.math.maxInt(OutputType)))
            else
                std.math.floatMax(OutputType);
            const clamped_result = std.math.clamp(scaled_result, min_val, max_val);

            // Store result
            const output_idx = (n * channels) + c;
            output.data[output_idx] = if (@typeInfo(OutputType) == .int)
                @as(OutputType, @intFromFloat(clamped_result))
            else
                @as(OutputType, clamped_result);
        }
    }
}

/// Calculate output shape for QLinearGlobalAveragePool
/// Output shape is (N, C, 1, 1, ...) where all spatial dimensions become 1
pub fn get_qlinearglobalaveragepool_output_shape(input_shape: []const usize) ![]usize {
    if (input_shape.len < 2) {
        return TensorMathError.InvalidDimensions;
    }

    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);

    // First two dimensions (batch_size, channels) remain the same
    output_shape[0] = input_shape[0]; // N (batch size)
    output_shape[1] = input_shape[1]; // C (channels)

    // All spatial dimensions become 1
    for (2..input_shape.len) |i| {
        output_shape[i] = 1;
    }

    return output_shape;
}

fn tryAcceleratedQLinearGlobalAveragePool(
    comptime InputType: type,
    x: anytype,
    x_scale: f32,
    x_zero_point: i32,
    output: anytype,
    y_scale: f32,
    y_zero_point: i32,
    batch_size: usize,
    channels: usize,
    spatial_size: usize,
) bool {
    // Allow DSP path on Cortex-M7 (no Helium) and Helium on N6
    if (!cmsis_dsp.is_enabled) return false;
    if (spatial_size == 0) return false;
    if (y_scale == 0.0) return false;
    if (x_scale == 0.0) return false;

    const total_channels = batch_size * channels;
    if (total_channels == 0) return false;
    if (spatial_size > std.math.maxInt(u32)) return false;

    const required_input = std.math.mul(usize, total_channels, spatial_size) catch return false;
    if (x.data.len < required_input) return false;
    if (output.data.len < total_channels) return false;

    const x_data_i8 = @as([*]const i8, @ptrCast(x.data.ptr));
    const x_data_u8 = @as([*]const u8, @ptrCast(x.data.ptr));

    // Fixed-point scale: scale_fp = x_scale / (y_scale * spatial_size)
    const scale_f32 = x_scale / (y_scale * @as(f32, @floatFromInt(spatial_size)));
    var shift: i32 = 30;
    var multiplier_i32: i32 = @as(i32, @intFromFloat(scale_f32 * 1073741824.0)); // 2^30
    // Normalize if overflowed
    while (multiplier_i32 >= 1073741824 and shift > 0) { // keep < 2^30
        multiplier_i32 >>= 1;
        shift -= 1;
    }

    const spatial_size_u32 = @as(u32, @intCast(spatial_size));
    const zp_term = @as(i32, @intCast(spatial_size)) * x_zero_point;

    var output_index: usize = 0;
    for (0..batch_size) |n| {
        for (0..channels) |c_idx| {
            const start = ((n * channels) + c_idx) * spatial_size;
            if (start + spatial_size > x.data.len) return false;
            if (output_index >= output.data.len) return false;

            // Compute mean in integer domain
            var mean_q7: i8 = 0;
            if (InputType == i8 and cmsis_dsp.supportsMeanQ7()) {
                cmsis_dsp.stats.arm_mean_q7(x_data_i8 + start, spatial_size_u32, &mean_q7);
                if (n == 0 and c_idx == 0) logDebug("[ZANT][qlinearglobalavgpool] path=i8+dsp mean\n", .{});
            } else if (InputType == u8 and cmsis_dsp.supportsMeanQ7()) {
                // shift u8 -> i8: (x - 128), mean back-conversion later
                var tmp: []i8 = pkg_allocator.alloc(i8, spatial_size) catch &[_]i8{};
                if (tmp.len == spatial_size) {
                    var j: usize = 0;
                    while (j < spatial_size) : (j += 1) tmp[j] = @as(i8, @intCast(@as(i32, x_data_u8[start + j]) - 128));
                    cmsis_dsp.stats.arm_mean_q7(@as([*]const i8, @ptrCast(tmp.ptr)), spatial_size_u32, &mean_q7);
                    pkg_allocator.free(tmp);
                    if (n == 0 and c_idx == 0) logDebug("[ZANT][qlinearglobalavgpool] path=u8+dsp mean via shift-128\n", .{});
                } else {
                    // fallback manual mean (u8)
                    var acc_u32: u32 = 0;
                    for (0..spatial_size) |i| acc_u32 += x_data_u8[start + i];
                    // mean_u8: (acc_u32 / spatial_size), then shift to q7 by -128
                    const mean_u8_i32: i32 = @as(i32, @intCast(@divTrunc(acc_u32, @as(u32, @intCast(spatial_size)))));
                    mean_q7 = @as(i8, @intCast(mean_u8_i32 - 128));
                    if (n == 0 and c_idx == 0) logDebug("[ZANT][qlinearglobalavgpool] path=u8+manual mean (alloc fail)\n", .{});
                }
            } else {
                // Manual mean (i8)
                if (InputType == i8) {
                    var acc_i32: i32 = 0;
                    for (0..spatial_size) |i| acc_i32 += x_data_i8[start + i];
                    mean_q7 = @as(i8, @intCast(@divTrunc(acc_i32, @as(i32, @intCast(spatial_size)))));
                    if (n == 0 and c_idx == 0) logDebug("[ZANT][qlinearglobalavgpool] path=i8+manual mean\n", .{});
                } else {
                    var acc_u32_m: u32 = 0;
                    for (0..spatial_size) |i| acc_u32_m += x_data_u8[start + i];
                    const mean_u8_i32_m: i32 = @as(i32, @intCast(@divTrunc(acc_u32_m, @as(u32, @intCast(spatial_size)))));
                    mean_q7 = @as(i8, @intCast(mean_u8_i32_m - 128));
                    if (n == 0 and c_idx == 0) logDebug("[ZANT][qlinearglobalavgpool] path=u8+manual mean\n", .{});
                }
            }

            // Now mean_q7 is the average in signed q7 domain of (x - x_zp), but we need to adjust zp
            // Compute adjusted sum equivalent: (mean_q7 * spatial_size) - zp_term, but we can scale directly from mean
            const mean_minus_zp: i32 = @as(i32, mean_q7) * @as(i32, @intCast(spatial_size)) - zp_term;
            const prod: i64 = @as(i64, mean_minus_zp) * @as(i64, multiplier_i32);
            const rounding: i64 = @as(i64, 1) << @as(u6, @intCast(@max(0, shift - 1)));
            const scaled_i32: i32 = @as(i32, @intCast((prod + rounding) >> @as(u6, @intCast(shift))));
            var with_zp: i32 = scaled_i32 + y_zero_point;

            const OutT = @TypeOf(output.data[0]);
            const out_min_i32: i32 = @as(i32, @intCast(std.math.minInt(OutT)));
            const out_max_i32: i32 = @as(i32, @intCast(std.math.maxInt(OutT)));
            if (with_zp < out_min_i32) with_zp = out_min_i32;
            if (with_zp > out_max_i32) with_zp = out_max_i32;

            output.data[output_index] = @as(OutT, @intCast(with_zp));
            output_index += 1;
        }
    }

    accelerators.markCmsisUsed();
    return true;
}
