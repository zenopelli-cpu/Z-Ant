const std = @import("std");

test {
    std.testing.log_level = .info;
    comptime {
        _ = @import("fromTensorProtoToTensor.zig");
        //_ = @import("graph_init.zig");
    }
}
