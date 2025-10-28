const std = @import("std");

test {
    _ = @import("Tensor/test_tensor.zig");
    _ = @import("Tensor/TensorMath/test_op_conv_relu.zig");
}
