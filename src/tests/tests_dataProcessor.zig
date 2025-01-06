const std = @import("std");
const DataProc = @import("dataprocessor");
const NormalizType = @import("dataprocessor").NormalizationType;
const Tensor = @import("tensor").Tensor;
const pkgAllocator = @import("pkgAllocator");

test "normalize float" {
    std.debug.print("\n     test: normalize float", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, -2.0 },
        [_]f32{ -4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    for (t1.data) |*val| {
        try std.testing.expect(1.0 >= val.*);
        try std.testing.expect(0.0 <= val.*);
    }
}

test "normalize float all different" {
    std.debug.print("\n     test: normalize float", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 3.0, 10.0 },
        [_]f32{ 4.0, 5.0, 1.0, 2.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 4 }; // 2x4 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    t1.info();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    t1.info();

    try std.testing.expect(1.0 - t1.data[3] < 0.001);
    try std.testing.expect(1.0 - t1.data[5] < 0.001);

    for (t1.data) |*val| {
        try std.testing.expect(1.0 >= val.*);
        try std.testing.expect(0.0 <= val.*);
    }
}

test "normalize delta 0 " {
    std.debug.print("\n     test: normalize delta 0", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 1.0 },
        [_]f32{ 0.0, 0.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    for (t1.data) |*val| {
        try std.testing.expect(0 == val.*);
    }
}

// Helper function to compute mean of a slice
fn compute_mean(comptime T: anytype, data: []const T) f64 {
    var sum: f64 = 0.0;
    for (data) |val| {
        sum += @as(f64, val);
    }
    return sum / @as(f64, @floatFromInt(data.len));
}

// Helper function to compute variance of a slice given its mean
fn compute_variance(comptime T: anytype, data: []const T, mean: f64) f64 {
    var sum_sq_diff: f64 = 0.0;
    for (data) |val| {
        const diff = @as(f64, val) - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / @as(f64, @floatFromInt(data.len));
}

test "standard normalize float 2D" {
    std.debug.print("\n     test: standard normalize float 2D", .{});

    const allocator = pkgAllocator.allocator;

    // Define a 2x2 tensor with diverse float values
    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    // Perform standard normalization
    try DataProc.normalize(f32, &t1, NormalizType.StandardDeviationNormalization);

    // Check each slice (since it's 2D, each row is a slice)
    for (0..t1.shape[0]) |b| {
        const slice_offset = b * t1.shape[1];
        const slice_size = t1.shape[1];
        const slice_data = t1.data[slice_offset .. slice_offset + slice_size];

        // Compute mean and variance
        var sum: f64 = 0.0;
        for (slice_data) |val| {
            sum += @as(f64, val);
        }
        const mean = sum / @as(f64, @floatFromInt(slice_size));

        var sum_sq_diff: f64 = 0.0;
        for (slice_data) |val| {
            const diff = @as(f64, val) - mean;
            sum_sq_diff += diff * diff;
        }

        // Expect mean close to 0 and variance close to 1
        var normalized_sum: f64 = 0.0;
        var normalized_sq_sum: f64 = 0.0;
        for (slice_data) |val| {
            normalized_sum += @as(f64, val);
            normalized_sq_sum += @as(f64, val) * @as(f64, val);
        }
        const normalized_mean = normalized_sum / @as(f64, @floatFromInt(slice_size));
        const normalized_variance = normalized_sq_sum / @as(f64, @floatFromInt(slice_size));

        try std.testing.expect(@abs(normalized_mean) < 1e-4);
        try std.testing.expect(@abs(normalized_variance - 1.0) < 1e-4);
    }
}
