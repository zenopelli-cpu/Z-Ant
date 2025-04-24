const std = @import("std");

test {
    std.testing.log_level = .info;

    comptime {
        _ = @import("Core/Tensor/TensorMath/test_op_convolution.zig");
        _ = @import("Core/test_core.zig");
        _ = @import("Utils/test_utils.zig");
    }
}
