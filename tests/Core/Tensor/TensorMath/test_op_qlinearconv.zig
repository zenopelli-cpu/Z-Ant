const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensor_math = zant.core.tensor.math_standard;

test "qlinearconv embedded rounding matches float path for negative accumulations" {
    const allocator = std.testing.allocator;

    var input_shape = [_]usize{ 1, 1, 1, 1 };
    var input_data: [1][1][1][1]i8 = .{.{.{.{1}}}};
    var input = try Tensor(i8).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var weight_shape = [_]usize{ 1, 1, 1, 1 };
    var weight_data: [1][1][1][1]i8 = .{.{.{.{-1}}}};
    var weight = try Tensor(i8).fromArray(&allocator, &weight_data, &weight_shape);
    defer weight.deinit();

    var output_shape = [_]usize{ 1, 1, 1, 1 };
    var embedded_output = try Tensor(i8).fromShape(&allocator, &output_shape);
    defer embedded_output.deinit();
    var reference_output = try Tensor(i8).fromShape(&allocator, &output_shape);
    defer reference_output.deinit();

    var scalar_shape = [_]usize{1};

    var x_scale_data: [1]f32 = .{1.0};
    var x_scale = try Tensor(f32).fromArray(&allocator, &x_scale_data, &scalar_shape);
    defer x_scale.deinit();

    var w_scale_data: [1]f32 = .{1.0};
    var w_scale = try Tensor(f32).fromArray(&allocator, &w_scale_data, &scalar_shape);
    defer w_scale.deinit();

    var y_scale_data: [1]f32 = .{1.0};
    var y_scale = try Tensor(f32).fromArray(&allocator, &y_scale_data, &scalar_shape);
    defer y_scale.deinit();

    var x_zp_data: [1]i8 = .{0};
    var x_zp = try Tensor(i8).fromArray(&allocator, &x_zp_data, &scalar_shape);
    defer x_zp.deinit();

    var w_zp_data: [1]i8 = .{0};
    var w_zp = try Tensor(i8).fromArray(&allocator, &w_zp_data, &scalar_shape);
    defer w_zp.deinit();

    var y_zp_data: [1]i8 = .{0};
    var y_zp = try Tensor(i8).fromArray(&allocator, &y_zp_data, &scalar_shape);
    defer y_zp.deinit();

    var bias_shape = [_]usize{1};
    var bias_data: [1]f32 = .{-0.5};
    var bias = try Tensor(f32).fromArray(&allocator, &bias_data, &bias_shape);
    defer bias.deinit();

    const auto_pad: []const u8 = "NOTSET";

    try tensor_math.qlinearconv_embedded_lean(
        i8,
        i8,
        f32,
        void,
        f32,
        &input,
        &x_scale,
        &x_zp,
        &weight,
        &w_scale,
        &w_zp,
        &embedded_output,
        &y_scale,
        &y_zp,
        &bias,
        null,
        null,
        null,
        null,
        auto_pad,
    );

    try tensor_math.qlinearconv_lean(
        i8,
        i8,
        f32,
        void,
        f32,
        &input,
        &x_scale,
        &x_zp,
        &weight,
        &w_scale,
        &w_zp,
        &reference_output,
        &y_scale,
        &y_zp,
        &bias,
        null,
        null,
        null,
        null,
        auto_pad,
    );

    const x_real = (@as(f32, @floatFromInt(input_data[0][0][0][0])) - @as(f32, @floatFromInt(x_zp_data[0]))) * x_scale_data[0];
    const w_real = (@as(f32, @floatFromInt(weight_data[0][0][0][0])) - @as(f32, @floatFromInt(w_zp_data[0]))) * w_scale_data[0];
    const expected_real = x_real * w_real + bias_data[0];
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), expected_real, 1e-6);

    const expected_q = @as(i8, @intFromFloat(@round(@as(f32, -1.5))));
    try std.testing.expectEqual(expected_q, embedded_output.data[0]);
    try std.testing.expectEqual(reference_output.data[0], embedded_output.data[0]);
}

fn run1x1SimdVsScalar(stride_hw: usize) !void {
    const allocator = std.testing.allocator;
    const auto_pad: []const u8 = "NOTSET";

    const groups: usize = 2;
    const group_in_channels: usize = 16;
    const group_out_channels: usize = 16;
    const in_channels = groups * group_in_channels;
    const out_channels = groups * group_out_channels;
    const in_height: usize = 5;
    const in_width: usize = 7;
    const out_height = @divFloor(in_height - 1, stride_hw) + 1;
    const out_width = @divFloor(in_width - 1, stride_hw) + 1;

    var input_shape = [_]usize{ 1, in_channels, in_height, in_width };
    var output_shape = [_]usize{ 1, out_channels, out_height, out_width };
    var weight_shape = [_]usize{ out_channels, group_in_channels, 1, 1 };
    var bias_shape = [_]usize{ out_channels };
    var scalar_shape = [_]usize{ 1 };
    var per_channel_shape = [_]usize{ out_channels };

    var input_i8 = try Tensor(i8).fromShape(&allocator, &input_shape);
    defer input_i8.deinit();
    for (input_i8.data, 0..) |*val, idx| {
        const base = @as(i32, @intCast(idx)) * 5 + 13;
        const mod = @mod(base, 255);
        val.* = @as(i8, @intCast(mod - 128));
    }

    var input_i16 = try Tensor(i16).fromShape(&allocator, &input_shape);
    defer input_i16.deinit();
    for (input_i16.data, 0..) |*val, idx| {
        val.* = @as(i16, input_i8.data[idx]);
    }

    var weight = try Tensor(i8).fromShape(&allocator, &weight_shape);
    defer weight.deinit();
    for (weight.data, 0..) |*val, idx| {
        const base = @as(i32, @intCast(idx)) * 7 + 5;
        const mod = @mod(base, 255);
        val.* = @as(i8, @intCast(mod - 128));
    }

    var bias = try Tensor(f32).fromShape(&allocator, &bias_shape);
    defer bias.deinit();
    for (bias.data, 0..) |*val, idx| {
        val.* = -1.5 + 0.05 * @as(f32, @floatFromInt(idx));
    }

    var w_scale = try Tensor(f32).fromShape(&allocator, &per_channel_shape);
    defer w_scale.deinit();
    for (w_scale.data, 0..) |*val, idx| {
        val.* = 0.5 + 0.01 * @as(f32, @floatFromInt(idx));
    }

    var w_zp = try Tensor(i8).fromShape(&allocator, &per_channel_shape);
    defer w_zp.deinit();
    for (w_zp.data, 0..) |*val, idx| {
        const pattern = @mod(@as(i32, @intCast(idx)) * 3, 11) - 5;
        val.* = @as(i8, @intCast(pattern));
    }

    var x_scale_data = [_]f32{0.125};
    var x_scale = try Tensor(f32).fromArray(&allocator, &x_scale_data, &scalar_shape);
    defer x_scale.deinit();

    var y_scale_data = [_]f32{0.03125};
    var y_scale = try Tensor(f32).fromArray(&allocator, &y_scale_data, &scalar_shape);
    defer y_scale.deinit();

    var x_zp_i8_data = [_]i8{17};
    var x_zp_i8 = try Tensor(i8).fromArray(&allocator, &x_zp_i8_data, &scalar_shape);
    defer x_zp_i8.deinit();

    var x_zp_i16_data = [_]i16{17};
    var x_zp_i16 = try Tensor(i16).fromArray(&allocator, &x_zp_i16_data, &scalar_shape);
    defer x_zp_i16.deinit();

    var y_zp_i8_data = [_]i8{-12};
    var y_zp_i8 = try Tensor(i8).fromArray(&allocator, &y_zp_i8_data, &scalar_shape);
    defer y_zp_i8.deinit();

    var y_zp_i16_data = [_]i16{-12};
    var y_zp_i16 = try Tensor(i16).fromArray(&allocator, &y_zp_i16_data, &scalar_shape);
    defer y_zp_i16.deinit();

    var output_i8 = try Tensor(i8).fromShape(&allocator, &output_shape);
    defer output_i8.deinit();

    var output_i16 = try Tensor(i16).fromShape(&allocator, &output_shape);
    defer output_i16.deinit();

    const stride_arr = [_]usize{ stride_hw, stride_hw };
    const stride_slice = stride_arr[0..];

    try tensor_math.qlinearconv_embedded_lean(
        i8,
        i8,
        f32,
        void,
        f32,
        &input_i8,
        &x_scale,
        &x_zp_i8,
        &weight,
        &w_scale,
        &w_zp,
        &output_i8,
        &y_scale,
        &y_zp_i8,
        &bias,
        stride_slice,
        null,
        null,
        groups,
        auto_pad,
    );

    try tensor_math.qlinearconv_embedded_lean(
        i16,
        i8,
        f32,
        void,
        f32,
        &input_i16,
        &x_scale,
        &x_zp_i16,
        &weight,
        &w_scale,
        &w_zp,
        &output_i16,
        &y_scale,
        &y_zp_i16,
        &bias,
        stride_slice,
        null,
        null,
        groups,
        auto_pad,
    );

    for (output_i8.data, 0..) |val, idx| {
        const simd_val = @as(i32, @intCast(val));
        const scalar_val = @as(i32, @intCast(output_i16.data[idx]));
        try std.testing.expectEqual(simd_val, scalar_val);
    }
}

test "qlinearconv 1x1 simd matches scalar with stride 1 and groups 2" {
    try run1x1SimdVsScalar(1);
}

test "qlinearconv 1x1 simd matches scalar with stride 2 and groups 2" {
    try run1x1SimdVsScalar(2);
}
