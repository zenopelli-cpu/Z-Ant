const std = @import("std");

comptime {
    _ = @import("Core/Tensor/TensorMath/test_op_convolution.zig");
    _ = @import("Core/test_core.zig");
    _ = @import("Utils/test_utils.zig");
    _ = @import("ImageToTensor/jpeg/test_jpeg_decoder.zig");
    _ = @import("ImageToTensor/test_utils.zig");
}
