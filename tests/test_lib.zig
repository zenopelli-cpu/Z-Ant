const std = @import("std");

test {
    _ = @import("Core/test_core.zig");
    _ = @import("DataHandler/test_dataHandler.zig");
    _ = @import("Model/test_model.zig");
    _ = @import("Utils/test_utils.zig");
}
