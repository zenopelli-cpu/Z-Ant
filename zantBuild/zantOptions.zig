const std = @import("std");

const stm32n6 = @import("stm32n6_flags.zig");
const testing = @import("testing_flags.zig");
const codegen = @import("codegen_flags.zig");

pub const ZantOptions = struct {
    stm32n6_flags: stm32n6.Stm32n6_flags,
    testing_flags: testing.Testing_flags,
    codegen_flags: codegen.Codegen_flags,

    pub fn init(b: *std.Build) !ZantOptions {
        return ZantOptions{
            .stm32n6_flags = try stm32n6.Stm32n6_flags.init(b),
            .testing_flags = try testing.Testing_flags.init(b),
            .codegen_flags = try codegen.Codegen_flags.init(b),
        };
    }
};
