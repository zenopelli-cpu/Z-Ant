const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensor_math = zant.core.tensor.math_standard;
const pkgAllocator = zant.utils.allocator;

test "QLinearConv depthwise uses SIMD lanes and matches scalar reference" {
    const allocator = pkgAllocator.allocator;
    const channels: usize = 3;
    const height: usize = 3;
    const width: usize = 3;

    var input_shape = [_]usize{ 1, channels, height, width };
    var input_data: [1][channels][height][width]i8 = .{
        .{
            .{ .{ 1, 2, 3 }, .{ 4, 5, 6 }, .{ 7, 8, 9 } },
            .{ .{ -1, -2, -3 }, .{ -4, -5, -6 }, .{ -7, -8, -9 } },
            .{ .{ 1, 0, -1 }, .{ 0, 1, 0 }, .{ -1, 0, 1 } },
        },
    };

    var weight_shape = [_]usize{ channels, 1, 3, 3 };
    var weight_data: [channels][1][3][3]i8 = .{
        .{ .{ .{ 1, 0, -1 }, .{ 2, 0, -2 }, .{ 1, 0, -1 } } },
        .{ .{ .{ -1, 0, 1 }, .{ -2, 0, 2 }, .{ -1, 0, 1 } } },
        .{ .{ .{ 0, 1, 0 }, .{ 1, 0, 1 }, .{ 0, 1, 0 } } },
    };

    var output_shape = [_]usize{ 1, channels, 1, 1 };

    var input_tensor = try Tensor(i8).fromArray(&allocator, &input_data, &input_shape);
    defer input_tensor.deinit();

    var weight_tensor = try Tensor(i8).fromArray(&allocator, &weight_data, &weight_shape);
    defer weight_tensor.deinit();

    var output_tensor = try Tensor(i8).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    var scale_shape = [_]usize{ 1 };
    var x_scale_data = [_]f32{ 1.0 };
    var y_scale_data = [_]f32{ 1.0 };

    var x_scale = try Tensor(f32).fromArray(&allocator, &x_scale_data, &scale_shape);
    defer x_scale.deinit();

    var y_scale = try Tensor(f32).fromArray(&allocator, &y_scale_data, &scale_shape);
    defer y_scale.deinit();

    var w_scale_shape = [_]usize{ channels };
    var w_scale_data: [channels]f32 = .{ 1.0, 1.0, 1.0 };
    var w_scale = try Tensor(f32).fromArray(&allocator, &w_scale_data, &w_scale_shape);
    defer w_scale.deinit();

    var zero_shape = [_]usize{ 1 };
    var x_zp_data = [_]i8{ 0 };
    var y_zp_data = [_]i8{ 0 };

    var x_zp = try Tensor(i8).fromArray(&allocator, &x_zp_data, &zero_shape);
    defer x_zp.deinit();

    var y_zp = try Tensor(i8).fromArray(&allocator, &y_zp_data, &zero_shape);
    defer y_zp.deinit();

    var w_zp_shape = [_]usize{ channels };
    var w_zp_data: [channels]i8 = .{ 0, 0, 0 };
    var w_zp = try Tensor(i8).fromArray(&allocator, &w_zp_data, &w_zp_shape);
    defer w_zp.deinit();

    var expected: [channels]i8 = undefined;
    for (0..channels) |c| {
        var acc: i32 = 0;
        var kh: usize = 0;
        while (kh < 3) : (kh += 1) {
            var kw: usize = 0;
            while (kw < 3) : (kw += 1) {
                const input_val = input_data[0][c][kh][kw];
                const weight_val = weight_data[c][0][kh][kw];
                const x_diff = @as(i32, input_val) - @as(i32, x_zp_data[0]);
                const w_diff = @as(i32, weight_val) - @as(i32, w_zp_data[c]);
                acc += x_diff * w_diff;
            }
        }
        acc = std.math.clamp(acc, std.math.minInt(i8), std.math.maxInt(i8));
        expected[c] = @as(i8, @intCast(acc));
    }

    tensor_math.op_qlinearconv.simd_debug_call_counter = 0;

    try tensor_math.qlinearconv_embedded_lean(
        i8,
        i8,
        f32,
        void,
        i32,
        &input_tensor,
        &x_scale,
        &x_zp,
        &weight_tensor,
        &w_scale,
        &w_zp,
        &output_tensor,
        &y_scale,
        &y_zp,
        null,
        null,
        null,
        null,
        channels,
        "NOTSET",
    );

    const simd_calls = tensor_math.op_qlinearconv.simd_debug_call_counter;
    try std.testing.expect(simd_calls > 0);

    try std.testing.expectEqual(@as(usize, channels), output_tensor.data.len);
    for (output_tensor.data, 0..) |got, idx| {
        try std.testing.expectEqual(expected[idx], got);
    }
}
