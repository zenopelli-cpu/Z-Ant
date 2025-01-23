const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test {
    _ = @import("test_lib_elementWise_math.zig");
    _ = @import("test_lib_logical_math.zig");
    _ = @import("test_lib_reduction_math.zig");
    _ = @import("test_lib_shape_math.zig");
    _ = @import("test_op_convolution.zig");
    _ = @import("test_op_dot_product.zig");
    _ = @import("test_op_pooling.zig");
}
