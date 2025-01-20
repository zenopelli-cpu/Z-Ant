const std = @import("std");

test {
    _ = @import("dataLoader.zig");
    _ = @import("dataProcessor.zig");
    // _ = @import("trainer.zig"); // This test is added as a separate test in tests/Trainer/trainer.zig
}
