const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test {
    _ = @import("lib_elementWise_math.zig");
    _ = @import("lib_logical_math.zig");
    _ = @import("lib_reduction_math.zig");
    _ = @import("lib_shape_math.zig");
    _ = @import("op_convolution.zig");
    _ = @import("op_dot_product.zig");
    _ = @import("op_pooling.zig");
}
