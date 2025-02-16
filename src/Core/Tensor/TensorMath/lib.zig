const tensor_m = @import("tensor_m");
pub usingnamespace tensor_m;

const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const dot_product_tensor = @import("op_dot_product.zig").dot_product_tensor;

pub fn matmul_and_info() !void {
    // Create two tensors for matrix multiplication
    var shape1 = [_]usize{ 2, 3 };
    var shape2 = [_]usize{ 3, 2 };

    var t1 = try Tensor(f32).fromShape(&pkg_allocator, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromShape(&pkg_allocator, &shape2);
    defer t2.deinit();

    // Fill tensors with some values
    for (0..t1.size) |i| {
        t1.data[i] = @floatFromInt(i + 1);
    }
    for (0..t2.size) |i| {
        t2.data[i] = @floatFromInt(i + 1);
    }

    // Perform matrix multiplication
    var result = try dot_product_tensor(f32, f32, &t1, &t2);
    defer result.deinit();

    // Print tensor info
    result.info();
}
