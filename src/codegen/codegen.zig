const std = @import("std");
const zant = @import("zant");
const IR = @import("IR_zant");

pub const codegen_v1 = @import("cg_v1/codegen_v1.zig");
pub const codegen_v1_exe = @import("cg_v1/main.zig");
pub const codegen_v2 = @import("cg_v2/codegen_v2.zig");
pub const codegen_v2_exe = @import("cg_v2/main.zig");

// --- importing codegen options
pub const codegen_options = @import("codegen_options");
