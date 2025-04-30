const std = @import("std");

comptime {
    _ = @import("CodeGen/renderer/test_zig_renderer.zig");
    _ = @import("Core/Tensor/TensorMath/test_op_convolution.zig");
    _ = @import("Core/test_core.zig");
    _ = @import("Utils/test_utils.zig");
}
