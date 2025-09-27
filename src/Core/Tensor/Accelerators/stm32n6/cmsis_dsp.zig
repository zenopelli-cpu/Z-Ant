const build_options = @import("build_options");

const n6_accel = @hasDecl(build_options, "stm32n6_accel") and build_options.stm32n6_accel;
const n6_cmsis = @hasDecl(build_options, "stm32n6_use_cmsis") and build_options.stm32n6_use_cmsis;
const n6_force_native = @hasDecl(build_options, "stm32n6_force_native") and build_options.stm32n6_force_native;

const m7_accel = @hasDecl(build_options, "cortexm7_accel") and build_options.cortexm7_accel;
const m7_cmsis = @hasDecl(build_options, "cortexm7_use_cmsis") and build_options.cortexm7_use_cmsis;
const m7_force_native = @hasDecl(build_options, "cortexm7_force_native") and build_options.cortexm7_force_native;

pub const is_enabled = ((n6_accel and n6_cmsis and !n6_force_native) or (m7_accel and m7_cmsis and !m7_force_native));

pub const stats = if (is_enabled) struct {
    pub extern fn arm_sum_q7(
        src: [*]const i8,
        block_size: u32,
        result: *i32,
    ) callconv(.C) void;

    pub extern fn arm_mean_q7(
        src: [*]const i8,
        block_size: u32,
        result: *i8,
    ) callconv(.C) void;
} else struct {
    pub fn arm_sum_q7(
        src: [*]const i8,
        block_size: u32,
        result: *i32,
    ) callconv(.C) void {
        _ = src;
        _ = block_size;
        result.* = 0;
    }

    pub fn arm_mean_q7(
        src: [*]const i8,
        block_size: u32,
        result: *i8,
    ) callconv(.C) void {
        _ = src;
        _ = block_size;
        result.* = 0;
    }
};

pub inline fn supportsSumQ7() bool {
    return is_enabled;
}

pub inline fn supportsMeanQ7() bool {
    return is_enabled;
}
