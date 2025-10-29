const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_conv_sigmoid_mul);

// ---------------------------------------------------------------
// --------------- TESTs FOR CONV+SIGMOID+MUL --------------------
// ---------------------------------------------------------------
// Conv+Sigmoid+Mul implements Attention Gate / Squeeze-and-Excitation
// Pattern: output = sigmoid(conv_out) * conv_out

/// Helper function to calculate expected sigmoid value
fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

// Test 1: Basic functionality - all positive values
test "Conv_Sigmoid_Mul - basic positive values" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - basic positive values\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: all 1s
    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    // Kernel: all 1s (produces positive conv output)
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_sigmoid_mul(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Output shape: [1, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[3]);

    // Conv output is 4, sigmoid(4) ≈ 0.9820, result = 0.9820 * 4 ≈ 3.928
    const conv_val: f32 = 4.0;
    const expected = sigmoid(conv_val) * conv_val;

    for (result.data) |val| {
        try std.testing.expectApproxEqRel(expected, val, 1e-4);
    }

    tests_log.info("Basic positive values work correctly\n", .{});
}

// Test 2: Negative conv output (sigmoid should be small, result near zero)
test "Conv_Sigmoid_Mul - negative conv output" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - negative conv output\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input: all 1s
    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    // Kernel: all -1s (produces negative conv output)
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ -1, -1 },
                [_]f32{ -1, -1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_sigmoid_mul(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Conv output is -4, sigmoid(-4) ≈ 0.018, result = 0.018 * -4 ≈ -0.072
    const conv_val: f32 = -4.0;
    const expected = sigmoid(conv_val) * conv_val;

    for (result.data) |val| {
        try std.testing.expectApproxEqRel(expected, val, 1e-3);
    }

    tests_log.info("Negative conv output handled correctly\n", .{});
}

// Test 3: Zero conv output (sigmoid(0) = 0.5, result = 0)
test "Conv_Sigmoid_Mul - zero conv output" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - zero conv output\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input with mixed values
    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, -1, 1 },
                [_]f32{ -1, 1, -1 },
                [_]f32{ 1, -1, 1 },
            },
        },
    };

    // Kernel designed to produce zero sum
    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_sigmoid_mul(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Conv output is 0, sigmoid(0) = 0.5, result = 0.5 * 0 = 0
    for (result.data) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), val, 1e-6);
    }

    tests_log.info("Zero conv output handled correctly\n", .{});
}

// Test 4: With bias
test "Conv_Sigmoid_Mul - with positive bias" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - with positive bias\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    // Bias: +2
    var bias_shape: [1]usize = [_]usize{1};
    var biasArray: [1]f32 = [_]f32{2};

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    var bias_tensor = try Tensor(f32).fromArray(&allocator, &biasArray, &bias_shape);
    defer bias_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_sigmoid_mul(f32, &input_tensor, &kernel_tensor, &bias_tensor, &stride, &pads, null, null, null);
    defer result.deinit();

    // Conv + bias = 4 + 2 = 6
    const conv_val: f32 = 6.0;
    const expected = sigmoid(conv_val) * conv_val;

    for (result.data) |val| {
        try std.testing.expectApproxEqRel(expected, val, 1e-4);
    }

    tests_log.info("Positive bias handled correctly\n", .{});
}

// Test 5: Multi-channel input
test "Conv_Sigmoid_Mul - multi-channel input" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - multi-channel input\n", .{});

    const allocator = pkgAllocator.allocator;

    // 1 batch, 2 channels, 3x3
    var input_shape: [4]usize = [_]usize{ 1, 2, 3, 3 };
    var inputArray: [1][2][3][3]f32 = [_][2][3][3]f32{
        [_][3][3]f32{
            // Channel 1
            [_][3]f32{
                [_]f32{ 1, 2, 1 },
                [_]f32{ 2, 3, 2 },
                [_]f32{ 1, 2, 1 },
            },
            // Channel 2
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    // 1 filter, 2 channels, 2x2 kernel
    var kernel_shape: [4]usize = [_]usize{ 1, 2, 2, 2 };
    var kernelArray: [1][2][2][2]f32 = [_][2][2][2]f32{
        [_][2][2]f32{
            // Channel 1
            [_][2]f32{
                [_]f32{ 1, 0 },
                [_]f32{ 0, 1 },
            },
            // Channel 2
            [_][2]f32{
                [_]f32{ 1, 0 },
                [_]f32{ 0, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const pads = [_]usize{ 0, 0, 0, 0 };

    var result = try TensMath.conv_sigmoid_mul(f32, &input_tensor, &kernel_tensor, null, &stride, &pads, null, null, null);
    defer result.deinit();

    // Output shape: [1, 1, 2, 2]
    try std.testing.expectEqual(@as(usize, 1), result.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), result.shape[1]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 2), result.shape[3]);

    // Verify output is reasonable (positive values with attention)
    for (result.data) |val| {
        try std.testing.expect(val > 0);
    }

    tests_log.info("Multi-channel input works correctly\n", .{});
}

// Test 6: Lean version with pre-allocated output
test "Conv_Sigmoid_Mul - lean version" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - lean version\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var inputArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    // Pre-allocate output
    var output_shape = [_]usize{ 1, 1, 2, 2 };
    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    const stride = [_]usize{1};

    try TensMath.conv_sigmoid_mul_lean(f32, &input_tensor, &kernel_tensor, &output_tensor, null, &stride, null, null, null, null);

    const conv_val: f32 = 4.0;
    const expected = sigmoid(conv_val) * conv_val;

    for (output_tensor.data) |val| {
        try std.testing.expectApproxEqRel(expected, val, 1e-4);
    }

    tests_log.info("Lean version works correctly\n", .{});
}

// Test 7: SAME_UPPER padding
test "Conv_Sigmoid_Mul - SAME_UPPER padding" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - SAME_UPPER padding\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 1, 1, 4, 4 };
    var inputArray: [1][1][4][4]f32 = [_][1][4][4]f32{
        [_][4][4]f32{
            [_][4]f32{
                [_]f32{ 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 3, 3 };
    var kernelArray: [1][1][3][3]f32 = [_][1][3][3]f32{
        [_][3][3]f32{
            [_][3]f32{
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
                [_]f32{ 1, 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};
    const auto_pad = "SAME_UPPER";

    var result = try TensMath.conv_sigmoid_mul(f32, &input_tensor, &kernel_tensor, null, &stride, null, null, null, auto_pad);
    defer result.deinit();

    // Output shape should match input shape with SAME padding
    try std.testing.expectEqual(@as(usize, 4), result.shape[2]);
    try std.testing.expectEqual(@as(usize, 4), result.shape[3]);

    // All values should be positive (attention mechanism)
    for (result.data) |val| {
        try std.testing.expect(val > 0);
    }

    tests_log.info("SAME_UPPER padding works correctly\n", .{});
}

// Test 8: Stride = 2
test "Conv_Sigmoid_Mul - stride 2" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - stride 2\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 1, 1, 5, 5 };
    var input = try Tensor(f32).fromShape(&allocator, &input_shape);
    defer input.deinit();

    for (0..25) |i| {
        input.data[i] = 1.0;
    }

    var weight_shape = [_]usize{ 1, 1, 3, 3 };
    var weight = try Tensor(f32).fromShape(&allocator, &weight_shape);
    defer weight.deinit();

    for (0..9) |i| {
        weight.data[i] = 1.0;
    }

    const stride_vals = [_]usize{ 2, 2 };

    var result = try TensMath.conv_sigmoid_mul(f32, &input, &weight, null, &stride_vals, null, null, null, null);
    defer result.deinit();

    // With stride=2, output should be [1, 1, 2, 2]
    try std.testing.expect(result.shape[2] == 2);
    try std.testing.expect(result.shape[3] == 2);

    tests_log.info("Stride 2 works correctly\n", .{});
}

// Test 9: Attention behavior (should suppress small values)
test "Conv_Sigmoid_Mul - attention behavior" {
    tests_log.info("\n     test: Conv_Sigmoid_Mul - attention behavior\n", .{});

    const allocator = pkgAllocator.allocator;

    // Create input with varying magnitudes
    var input_shape: [4]usize = [_]usize{ 1, 1, 4, 4 };
    var inputArray: [1][1][4][4]f32 = [_][1][4][4]f32{
        [_][4][4]f32{
            [_][4]f32{
                [_]f32{ 5, 5, 1, 1 },
                [_]f32{ 5, 5, 1, 1 },
                [_]f32{ 1, 1, 1, 1 },
                [_]f32{ 1, 1, 1, 1 },
            },
        },
    };

    var kernel_shape: [4]usize = [_]usize{ 1, 1, 2, 2 };
    var kernelArray: [1][1][2][2]f32 = [_][1][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1, 1 },
                [_]f32{ 1, 1 },
            },
        },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    const stride = [_]usize{1};

    var result = try TensMath.conv_sigmoid_mul(f32, &input_tensor, &kernel_tensor, null, &stride, null, null, null, null);
    defer result.deinit();

    // Top-left region (high values) should have higher output than bottom-right (low values)
    // This demonstrates the attention mechanism
    const top_left = result.data[0]; // High input region
    const bottom_right = result.data[result.data.len - 1]; // Low input region

    try std.testing.expect(top_left > bottom_right);

    tests_log.info("Attention behavior verified\n", .{});
}
