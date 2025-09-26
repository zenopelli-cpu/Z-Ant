const zant = @import("../../../zant.zig");
const build_options = @import("build_options");

const TensorModule = zant.core.tensor;

// Flags (mirroring STM32N6 style but for Cortex-M7)
const force_native = @hasDecl(build_options, "cortexm7_force_native") and build_options.cortexm7_force_native;
const use_cmsis = @hasDecl(build_options, "cortexm7_use_cmsis") and build_options.cortexm7_use_cmsis;

// Simple test/telemetry hooks fully in Zig (no C stub for M7)
var g_cmsis_used: bool = false;

pub fn markCmsisUsed() void {
    g_cmsis_used = true;
}

pub fn cmsisUsed() bool {
    return g_cmsis_used;
}

pub fn resetTestHooks() void {
    g_cmsis_used = false;
}

// Optional backend entry points. We don't currently provide an f32 CMSIS path on M7.
// Keep the symbol absent so the generic path is used.
// pub fn tryConvLean(...) !bool { return false; }
