const std = @import("std");
const test_options = @import("test_options");
const test_name = test_options.test_name;

comptime {
    _ = @import("Core/test_core.zig");
    _ = @import("Utils/test_utils.zig");
    _ = @import("ImageToTensor/jpeg/test_jpeg_decoder.zig");
    _ = @import("ImageToTensor/test_utils.zig");
    // _ = @import("CodeGen//renderer//test_zig_renderer.zig"); <<<< to uncomment!!!
    _ = @import("IR_graph/IR_graph.zig");
}
