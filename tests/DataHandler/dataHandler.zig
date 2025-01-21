const std = @import("std");
const test_options = @import("test_options");

test {
    _ = @import("dataLoader.zig");
    _ = @import("dataProcessor.zig");

    if (test_options.heavy) {
        _ = @import("trainer.zig");
    }
}