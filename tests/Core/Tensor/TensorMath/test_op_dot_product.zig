const std = @import("std");
const pkgAllocator = @import("pkgAllocator");
const TensMath = @import("tensor_m");
const Tensor = @import("tensor").Tensor;
const TensorMathError = @import("errorHandler").TensorMathError;
const ErrorHandler = @import("errorHandler");

test "Dot product 2x2" {
    std.debug.print("\n     test:Dot product 2x2", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);

    var result_tensor = try TensMath.dot_product_tensor(f32, f64, &t1, &t2);

    try std.testing.expect(9.0 == result_tensor.data[0]);
    try std.testing.expect(12.0 == result_tensor.data[1]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible sizes for dot product" {
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(f32, f64, &t1, &t2));

    _ = TensMath.dot_product_tensor(f32, f64, &t1, &t2) catch |err| {
        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible shapes for dot product" {
    std.debug.print("\n     test: Error when input tensors have incompatible shapes for dot product", .{});
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 4x1 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "Compare dot product implementations with execution time" {
    std.debug.print("\nTest: Compare dot product implementations with execution time\n", .{});
    const allocator = pkgAllocator.allocator;

    // Create test tensors
    var shape: [2]usize = [_]usize{ 16, 16 };
    var matrix1: [16][16]f32 = undefined;
    var matrix2: [16][16]f32 = undefined;

    // Initialize with different values for each matrix
    for (0..16) |i| {
        for (0..16) |j| {
            matrix1[i][j] = @floatFromInt(i * 16 + j + 1);
            matrix2[i][j] = @floatFromInt((15 - i) * 16 + (15 - j) + 1); // Reverse pattern
        }
    }

    var t1 = try Tensor(f32).fromArray(&allocator, &matrix1, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &matrix2, &shape);
    defer t1.deinit();
    defer t2.deinit();

    // Run multiple iterations for more accurate timing
    const iterations = 100;
    var total_simd: i64 = 0;
    var total_flat: i64 = 0;

    for (0..iterations) |_| {
        const start_simd = std.time.nanoTimestamp();
        var result_simd = try TensMath.dot_product_tensor(f32, f32, &t1, &t2);
        const end_simd = std.time.nanoTimestamp();
        total_simd += @as(i64, @intCast(end_simd - start_simd));
        result_simd.deinit();

        const start_flat = std.time.nanoTimestamp();
        var result_flat = try TensMath.dot_product_tensor_flat(f32, f32, &t1, &t2);
        const end_flat = std.time.nanoTimestamp();
        total_flat += @as(i64, @intCast(end_flat - start_flat));
        result_flat.deinit();
    }

    const avg_simd = @divFloor(total_simd, iterations);
    const avg_flat = @divFloor(total_flat, iterations);

    std.debug.print("Average over {d} iterations:\n", .{iterations});
    std.debug.print("SIMD execution time: {d} ns\n", .{avg_simd});
    std.debug.print("Flat execution time: {d} ns\n", .{avg_flat});
    std.debug.print("SIMD is {d:.2}x {s}\n", .{ if (avg_simd < avg_flat)
        @as(f64, @floatFromInt(avg_flat)) / @as(f64, @floatFromInt(avg_simd))
    else
        @as(f64, @floatFromInt(avg_simd)) / @as(f64, @floatFromInt(avg_flat)), if (avg_simd < avg_flat) "faster" else "slower" });
}
