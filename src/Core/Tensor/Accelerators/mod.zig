const zant = @import("../../../zant.zig");
const build_options = @import("build_options");
const common = @import("common.zig");

const TensorModule = zant.core.tensor;

pub const AutoPadMode = common.AutoPadMode;
pub const ConvPreparedParams = common.ConvPreparedParams;

fn backendModule() type {
    if (@hasDecl(build_options, "stm32n6_accel") and build_options.stm32n6_accel) {
        return @import("stm32n6.zig");
    }
    if (@hasDecl(build_options, "cortexm7_accel") and build_options.cortexm7_accel) {
        return @import("cortexm7.zig");
    }
    return @import("null_accelerator.zig");
}

const Backend = backendModule();

pub fn isStm32n6Enabled() bool {
    return @hasDecl(build_options, "stm32n6_accel") and build_options.stm32n6_accel;
}

pub fn isCortexm7Enabled() bool {
    return @hasDecl(build_options, "cortexm7_accel") and build_options.cortexm7_accel;
}

pub fn isForceNativeEnabled() bool {
    return @hasDecl(build_options, "stm32n6_force_native") and build_options.stm32n6_force_native;
}

pub fn isForceNativeM7Enabled() bool {
    return @hasDecl(build_options, "cortexm7_force_native") and build_options.cortexm7_force_native;
}

pub fn isCmsisRequested() bool {
    return @hasDecl(build_options, "stm32n6_use_cmsis") and build_options.stm32n6_use_cmsis;
}

pub fn isCmsisRequestedM7() bool {
    return @hasDecl(build_options, "cortexm7_use_cmsis") and build_options.cortexm7_use_cmsis;
}

pub fn isEthosRequested() bool {
    return @hasDecl(build_options, "stm32n6_use_ethos") and build_options.stm32n6_use_ethos;
}

pub fn canUseCmsisHelium() bool {
    const n6 = isStm32n6Enabled() and !isForceNativeEnabled() and isCmsisRequested();
    const m7 = isCortexm7Enabled() and !isForceNativeM7Enabled() and isCmsisRequestedM7();
    return n6 or m7;
}

pub fn tryConvLean(
    comptime T: type,
    input: *const TensorModule.Tensor(T),
    weight: *const TensorModule.Tensor(T),
    output: *TensorModule.Tensor(T),
    bias: ?[]const T,
    params: ConvPreparedParams,
) !bool {
    if (@hasDecl(Backend, "tryConvLean")) {
        return try Backend.tryConvLean(T, input, weight, output, bias, params);
    }
    return false;
}

pub fn resetTestHooks() void {
    if (@hasDecl(Backend, "resetTestHooks")) {
        Backend.resetTestHooks();
    }
}

pub fn markCmsisUsed() void {
    if (@hasDecl(Backend, "markCmsisUsed")) {
        Backend.markCmsisUsed();
    }
}

pub fn cmsisUsed() bool {
    if (@hasDecl(Backend, "cmsisUsed")) {
        return Backend.cmsisUsed();
    }
    return false;
}

pub fn ethosUsed() bool {
    if (@hasDecl(Backend, "ethosUsed")) {
        return Backend.ethosUsed();
    }
    return false;
}
