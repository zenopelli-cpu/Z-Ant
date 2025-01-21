const std = @import("std");
const pkgAllocator = @import("pkgAllocator");
const TensMath = @import("tensor_m");
const Tensor = @import("tensor").Tensor;

test "mean" {
    std.debug.print("\n     test:mean", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [1]usize = [_]usize{3}; // 2x3 matrix
    var inputArray: [3]f32 = [_]f32{ 1.0, 2.0, 3.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);

    try std.testing.expect(2.0 == TensMath.mean(f32, &t1));

    t1.deinit();
}
