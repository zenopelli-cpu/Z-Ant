const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running quant_tensor_math tests\n", .{});
}

test {
    _ = @import("test_op_dequantize.zig");
    _ = @import("test_op_quantize.zig");
    _ = @import("test_op_pooling.zig");
    // _ = @import("test_quant_op_convolution.zig");
    _ = @import("test_quant_op_mat_mul.zig");
}
