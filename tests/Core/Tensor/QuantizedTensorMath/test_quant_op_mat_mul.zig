const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const QuantTensMath = zant.core.tensor.quantized_math;
const Tensor = zant.core.tensor.Tensor;
const tensorType = zant.core.tensor.TensorType;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const tensorDetails = zant.core.tensor.TensorDetails;
const quantDetails = zant.core.tensor.QuantDetails;
const ErrorHandler = zant.utils.error_handler;

test "QUANT MatMul 2x2" {
    std.debug.print("\n     test: QUANT MatMul 2x2", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    // prepare input tensors
    var inputArray: [2][2]i8 = [_][2]i8{
        [_]i8{ 1, 2 },
        [_]i8{ 4, 5 },
    };

    var t1 = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    t1.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 0.01,
            .zero_point = -2,
        },
    };
    defer t1.deinit();

    var t2 = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    t2.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 0.01,
            .zero_point = -2,
        },
    };
    defer t2.deinit();

    // compute multiplication
    var result_tensor = try QuantTensMath.quant_mat_mul(i8, &t1, &t2);
    defer result_tensor.deinit();

    // check details
    try std.testing.expect( result_tensor.details.quant.tensorType == tensorType.QuantTensor);
    try std.testing.expect( result_tensor.details.quant.scale_factor == 0.0001);
    try std.testing.expect( result_tensor.details.quant.zero_point == 0);

    // check results
    try std.testing.expect( 33 == result_tensor.data[0]);
    try std.testing.expect( 40 == result_tensor.data[1]);
    try std.testing.expect( 60 == result_tensor.data[2]);
    try std.testing.expect( 73 == result_tensor.data[3]);

}

test "QUANT Error when input tensors have incompatible sizes for MatMul" {
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(i8).fromShape(&allocator, &shape1);
    t1.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 1,
            .zero_point = 0,
        },
    };
    var t2 = try Tensor(i8).fromShape(&allocator, &shape2);
    t2.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 1,
            .zero_point = 0,
        },
    };

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, QuantTensMath.quant_mat_mul(i8, &t1, &t2));

    _ = QuantTensMath.quant_mat_mul(i8, &t1, &t2) catch |err| {
        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "QUANT Error when input tensors have incompatible shapes for MatMul" {
    std.debug.print("\n     test: QUANT Error when input tensors have incompatible shapes for MatMul", .{});
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 4x1 matrix
    var t1 = try Tensor(i8).fromShape(&allocator, &shape1);
    t1.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 1,
            .zero_point = 0,
        },
    };
    var t2 = try Tensor(i8).fromShape(&allocator, &shape2);
    t2.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 1,
            .zero_point = 0,
        },
    };

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, QuantTensMath.quant_mat_mul(i8, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "QUANT MatMul 4x4" {
    std.debug.print("\n     test: QUANT MatMul 4x4", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 4, 4 }; // 4x4 matrix

    // prepare input tensors
    var inputArray: [4][4]i8 = [_][4]i8{
        [_]i8{ 1, 2, 3, 4 },
        [_]i8{ 5, 6, 7, 8 },
        [_]i8{ 9, 10, 11, 12 },
        [_]i8{ 13, 14, 15, 16 },
    };

    var t1 = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    t1.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 0.03,
            .zero_point = 0,
        },
    };
    defer t1.deinit();

    var t2 = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    t2.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 0.02,
            .zero_point = 0,
        },
    };
    defer t2.deinit();

    // compute multiplication
    var result_tensor = try QuantTensMath.quant_mat_mul(i8, &t1, &t2);
    defer result_tensor.deinit();

    // check details
    try std.testing.expect( result_tensor.details.quant.tensorType == tensorType.QuantTensor);
    try std.testing.expect( result_tensor.details.quant.scale_factor < 0.00061 and result_tensor.details.quant.scale_factor > 0.00059);
    try std.testing.expect( result_tensor.details.quant.zero_point == 0);

    // check results
    try std.testing.expect( 90 == result_tensor.data[0]);
    try std.testing.expect( 100 == result_tensor.data[1]);
    try std.testing.expect( 110 == result_tensor.data[2]);
    try std.testing.expect( 120 == result_tensor.data[3]);
    try std.testing.expect( 127 == result_tensor.data[4]);
    try std.testing.expect( 127 == result_tensor.data[5]);
    try std.testing.expect( 127 == result_tensor.data[6]);
    try std.testing.expect( 127 == result_tensor.data[7]);
    try std.testing.expect( 127 == result_tensor.data[8]);
    try std.testing.expect( 127 == result_tensor.data[9]);
    try std.testing.expect( 127 == result_tensor.data[10]);
    try std.testing.expect( 127 == result_tensor.data[11]);
    try std.testing.expect( 127 == result_tensor.data[12]);
    try std.testing.expect( 127 == result_tensor.data[13]);
    try std.testing.expect( 127 == result_tensor.data[14]);
    try std.testing.expect( 127 == result_tensor.data[15]);

}




// test "Compare MatMul implementations with execution time" {
//     std.debug.print("\nTest: Compare MatMul implementations with execution time\n", .{});
//     const allocator = pkgAllocator.allocator;

//     // Create test tensors
//     var shape: [2]usize = [_]usize{ 16, 16 };
//     var matrix1: [16][16]f32 = undefined;
//     var matrix2: [16][16]f32 = undefined;

//     // Initialize with different values for each matrix
//     for (0..16) |i| {
//         for (0..16) |j| {
//             matrix1[i][j] = @floatFromInt(i * 16 + j + 1);
//             matrix2[i][j] = @floatFromInt((15 - i) * 16 + (15 - j) + 1); // Reverse pattern
//         }
//     }

//     var t1 = try Tensor(f32).fromArray(&allocator, &matrix1, &shape);
//     var t2 = try Tensor(f32).fromArray(&allocator, &matrix2, &shape);
//     defer t1.deinit();
//     defer t2.deinit();

//     // Run multiple iterations for more accurate timing
//     const iterations = 100;
//     var total_simd: i64 = 0;
//     var total_flat: i64 = 0;

//     for (0..iterations) |_| {
//         const start_simd = std.time.nanoTimestamp();
//         var result_simd = try TensMath.mat_mul(f32, &t1, &t2);
//         const end_simd = std.time.nanoTimestamp();
//         total_simd += @as(i64, @intCast(end_simd - start_simd));
//         result_simd.deinit();

//         const start_flat = std.time.nanoTimestamp();
//         var result_flat = try TensMath.dot_product_tensor_flat(f32, f32, &t1, &t2);
//         defer result_flat.deinit();
//         const end_flat = std.time.nanoTimestamp();
//         total_flat += @as(i64, @intCast(end_flat - start_flat));
//         result_flat.deinit();
//     }

//     const avg_simd = @divFloor(total_simd, iterations);
//     const avg_flat = @divFloor(total_flat, iterations);

//     std.debug.print("Average over {d} iterations:\n", .{iterations});
//     std.debug.print("SIMD execution time: {d} ns\n", .{avg_simd});
//     std.debug.print("Flat execution time: {d} ns\n", .{avg_flat});
//     std.debug.print("SIMD is {d:.2}x {s}\n", .{ if (avg_simd < avg_flat)
//         @as(f64, @floatFromInt(avg_flat)) / @as(f64, @floatFromInt(avg_simd))
//     else
//         @as(f64, @floatFromInt(avg_simd)) / @as(f64, @floatFromInt(avg_flat)), if (avg_simd < avg_flat) "faster" else "slower" });
// }
