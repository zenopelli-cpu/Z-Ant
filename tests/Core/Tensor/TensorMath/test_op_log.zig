const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const TensMath = zant.core.tensor.math_standard;
const TensorError = zant.utils.error_handler.TensorError;

const tests_log = std.log.scoped(.test_log);

test "log_standard - basic case" {
    tests_log.info("\n     test: log_standard - basic case", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{3};
    var inputArray = [_]f32{ 1.0, -2.0, 0.0 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    const expected = [_]f32{ 1.0, -0.86466472, 0.0 }; // exp(-2.0) - 1 ≈ -0.86466472
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-5);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, result.shape);
}

test "log_standard - different alpha" {
    tests_log.info("\n     test: log_standard - different alpha", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{3};
    var inputArray = [_]f32{ 1.0, -2.0, 0.0 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    const expected = [_]f32{ 1.0, -1.72932944, 0.0 }; // 2.0 * (exp(-2.0) - 1) ≈ -1.72932944
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-5);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, result.shape);
}

test "log_standard - single element" {
    tests_log.info("\n     test: log_standard - single element", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{1};
    var inputArray = [_]f32{-1.0};

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    const expected = [_]f32{-0.63212056}; // exp(-1.0) - 1 ≈ -0.63212056
    for (result.data, expected) |got, exp| {
        try std.testing.expectApproxEqRel(got, exp, 1e-5);
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{1}, result.shape);
}

test "log_standard - empty tensor" {
    tests_log.info("\n     test: log_standard - empty tensor", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{0};

    var input = try Tensor(f32).fromShape(&allocator, &shape);
    defer input.deinit();

    var result = try TensMath.log(f32, &input);
    defer result.deinit();

    try std.testing.expectEqual(result.data.len, 0);
    try std.testing.expectEqualSlices(usize, &[_]usize{0}, result.shape);
}

test "log_standard - invalid type" {
    tests_log.info("\n     test: log_standard - invalid type", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{3};
    var inputArray = [_]i32{ 1, -2, 0 };

    var input = try Tensor(i32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    try std.testing.expectError(TensorMathError.InvalidDataType, TensMath.log(i32, &input));
    _ = TensMath.log(i32, &input) catch |err| {
        tests_log.warn("\n     Error: {s}", .{zant.utils.error_handler.errorDetails(err)});
    };
}

test "log_standard - non 1D input" {
    tests_log.info("\n     test: log_standard - non 1D input", .{});
    const allocator = pkgAllocator.allocator;
    var shape = [_]usize{ 2, 2 };
    var inputArray = [_]f32{ 1.0, -2.0, 0.0, 1.0 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    try std.testing.expectError(TensorMathError.InvalidInput, TensMath.log(f32, &input, 1.0));
    _ = TensMath.log(f32, &input) catch |err| {
        tests_log.warn("\n     Error: {s}", .{zant.utils.error_handler.errorDetails(err)});
    };
}
