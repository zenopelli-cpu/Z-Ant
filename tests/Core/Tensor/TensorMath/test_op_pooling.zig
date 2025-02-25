const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const PoolingType = zant.model.layer.poolingLayer.PoolingType;

test "Pooling 2D" {
    std.debug.print("\n     test: Pooling 2D\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3 input, same as original
    var shape_tensor: [2]usize = [_]usize{ 3, 3 };
    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 40.0, 50.0, 60.0 },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    // Calculate W = number of windows
    const input_rows = input1.shape[0];
    const input_cols = input1.shape[1];
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; // 2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; // 2
    const W = out_rows * out_cols; // 4

    // Instead of used_input being shape of input, now [W,3,3]
    var used_input1_shape = [_]usize{ W, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output1: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output1.deinit();

    // Same checks for output as original
    try std.testing.expectEqual(output1.shape.len, input1.shape.len);
    try std.testing.expectEqual(output1.shape[0], 2);
    try std.testing.expectEqual(output1.shape[1], 2);

    try std.testing.expectEqual(output1.data[0], 5);
    try std.testing.expectEqual(output1.data[1], 6);
    try std.testing.expectEqual(output1.data[2], 50);
    try std.testing.expectEqual(output1.data[3], 60);

    // Now we have 4 windows (W=4), each 3x3
    // The original code expected certain pattern in used_input:
    // (1,1)=5 for first window
    // (1,2)=6 for second window
    // (2,1)=50 for third window
    // (2,2)=60 for fourth window
    //
    // Windows:
    // w=0: top-left max at (1,1)
    // w=1: top-right max at (1,2)
    // w=2: bottom-left max at (2,1)
    // w=3: bottom-right max at (2,2)

    // Check w=0 window:
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[0 * 9 + 1 * 3 + 1]);

    // w=1: top-right (1,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[1 * 9 + 1 * 3 + 2]);

    // w=2: bottom-left (2,1)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[2 * 9 + 2 * 3 + 1]);

    // w=3: bottom-right (2,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[3 * 9 + 2 * 3 + 2]);

    // Another test with kernel2 and stride2
    var kernel2: [2]usize = [2]usize{ 2, 2 };
    var stride2: [2]usize = [2]usize{ 2, 2 };

    const out_rows_2 = (3 - 2 + 1) / 2;
    const out_cols_2 = (3 - 2 + 1) / 2;
    const W2 = out_rows_2 * out_cols_2;

    var used_input2_shape = [_]usize{ W2, 3, 3 };
    var used_input2 = try Tensor(u8).fromShape(&allocator, used_input2_shape[0..]);
    defer used_input2.deinit();
    for (used_input2.data) |*val| val.* = 0;

    var output2: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input2, &kernel2, &stride2, PoolingType.Max);
    defer output2.deinit();

    try std.testing.expectEqual(output2.shape.len, input1.shape.len);
    try std.testing.expectEqual(output2.shape[0], 1);
    try std.testing.expectEqual(output2.shape[1], 1);
    try std.testing.expectEqual(output2.data[0], 5);

    // single window w=0 max at (1,1)
    try std.testing.expectEqual(@as(u1, 1), used_input2.data[0 * (3 * 3) + 1 * 3 + 1]);
}

test "Pooling multidim" {
    std.debug.print("\n     test: Pooling multidim\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3x3 input as original
    var shape_tensor: [3]usize = [_]usize{ 3, 3, 3 };
    var inputArray: [3][3][3]f32 = [_][3][3]f32{
        [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ 10.0, 20.0, 30.0 },
            [_]f32{ 40.0, 0.0, -10.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ -1.0, -2.0, -3.0 },
            [_]f32{ -4.0, -5.0, -6.0 },
            [_]f32{ -40.0, -50.0, -60.0 },
        },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    const d = input1.shape[0]; // 3
    const input_rows = input1.shape[1]; //3
    const input_cols = input1.shape[2]; //3
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; //2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; //2
    const W_tot = d * out_rows * out_cols; // 3*2*2=12

    var used_input1_shape = [_]usize{ W_tot, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output.deinit();

    try std.testing.expectEqual(output.shape.len, input1.shape.len);
    try std.testing.expectEqual(output.shape[0], 3);
    try std.testing.expectEqual(output.shape[1], 2);
    try std.testing.expectEqual(output.shape[2], 2);

    // We do not check used_input details here, just shapes as in original.
}
