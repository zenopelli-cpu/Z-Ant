const std = @import("std");

test {
    _ = @import("Tensor/test_tensor.zig");
    _ = @import("Tensor/TensorMath/test_op_conv_relu.zig");
    // _ = @import("Quantization/test_quantization.zig");
    // _ = @import("Quantization/test_clustering.zig");
}
