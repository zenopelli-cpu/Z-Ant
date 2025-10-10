const std = @import("std");

pub const Stm32n6_flags = struct {
    stm32n6_accel: bool,
    stm32n6_cmsis_path: []const u8,
    stm32n6_force_native: bool,
    stm32n6_use_cmsis: bool,
    stm32n6_use_ethos: bool,
    stm32n6_ethos_path: []const u8,

    pub fn init(b: *std.Build) !Stm32n6_flags {
        return Stm32n6_flags{
            .stm32n6_accel = b.option(bool, "stm32n6_accel", "Enable STM32 N6 accelerator support") orelse false,
            .stm32n6_cmsis_path = b.option([]const u8, "stm32n6_cmsis_path", "Optional CMSIS include path for STM32 N6 support") orelse "",
            .stm32n6_force_native = b.option(bool, "stm32n6_force_native", "Force STM32 N6 accelerator stubs on non-Thumb targets (useful for host testing)") orelse false,
            .stm32n6_use_cmsis = b.option(bool, "stm32n6_use_cmsis", "Enable CMSIS Helium kernels for STM32 N6") orelse false,
            .stm32n6_use_ethos = b.option(bool, "stm32n6_use_ethos", "Enable Ethos-U integration stubs for STM32 N6") orelse false,
            .stm32n6_ethos_path = b.option([]const u8, "stm32n6_ethos_path", "Optional include path for Ethos-U driver headers") orelse "",
        };
    }
};
