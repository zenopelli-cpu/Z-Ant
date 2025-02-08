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
    var shape: [2]usize = [_]usize{ 4, 4 };
    var inputArray: [4][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 3.0, 4.0 },
        [_]f32{ 5.0, 6.0, 7.0, 8.0 },
        [_]f32{ 9.0, 10.0, 11.0, 12.0 },
        [_]f32{ 13.0, 14.0, 15.0, 16.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    defer t2.deinit();

    const start_simd = std.time.nanoTimestamp();
    var result_simd = try TensMath.dot_product_tensor(f32, f32, &t1, &t2);
    const end_simd = std.time.nanoTimestamp();
    const duration_simd = end_simd - start_simd;
    defer result_simd.deinit();

    const start_flat = std.time.nanoTimestamp();
    var result_flat = try TensMath.dot_product_tensor_flat(f32, f32, &t1, &t2);
    const end_flat = std.time.nanoTimestamp();
    const duration_flat = end_flat - start_flat;
    defer result_flat.deinit();

    std.debug.print("SIMD execution time: {d} ns\n", .{duration_simd});
    std.debug.print("Flat execution time: {d} ns\n", .{duration_flat});

    // Compare results
    for (result_simd.data, result_flat.data) |v1, v2| {
        try std.testing.expectApproxEqAbs(v1, v2, 0.001);
    }
}
