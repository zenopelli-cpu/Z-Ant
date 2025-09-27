const build_options = @import("build_options");

// Generalized enablement: true if either STM32N6+CMSIS is enabled or Cortex-M7+CMSIS is enabled
const n6_accel = @hasDecl(build_options, "stm32n6_accel") and build_options.stm32n6_accel;
const n6_cmsis = @hasDecl(build_options, "stm32n6_use_cmsis") and build_options.stm32n6_use_cmsis;
const n6_force_native = @hasDecl(build_options, "stm32n6_force_native") and build_options.stm32n6_force_native;

const m7_accel = @hasDecl(build_options, "cortexm7_accel") and build_options.cortexm7_accel;
const m7_cmsis = @hasDecl(build_options, "cortexm7_use_cmsis") and build_options.cortexm7_use_cmsis;
const m7_force_native = @hasDecl(build_options, "cortexm7_force_native") and build_options.cortexm7_force_native;

/// Indicates whether CMSIS-NN kernels are enabled for the current build (N6 Helium or plain M7)
pub const is_enabled = ((n6_accel and n6_cmsis and !n6_force_native) or (m7_accel and m7_cmsis and !m7_force_native));

/// Common CMSIS-NN structures mirrored from the C headers.
/// Keeping them in a shared module makes it easier to add new Helium bindings
/// without copy/pasting the extern declarations across operators.
pub const Dims = extern struct {
    n: i32,
    h: i32,
    w: i32,
    c: i32,
};

pub const Context = extern struct {
    buf: ?*anyopaque,
    size: i32,
};

pub const ConvParams = extern struct {
    input_offset: i32,
    output_offset: i32,
    stride: extern struct { h: i32, w: i32 },
    padding: extern struct { h: i32, w: i32 },
    dilation: extern struct { h: i32, w: i32 },
    activation: extern struct { min: i32, max: i32 },
};

pub const PerChannelQuantParams = extern struct {
    multiplier: [*]i32,
    shift: [*]i32,
};

pub const DwConvParams = extern struct {
    input_offset: i32,
    output_offset: i32,
    ch_mult: i32,
    stride: extern struct { h: i32, w: i32 },
    padding: extern struct { h: i32, w: i32 },
    dilation: extern struct { h: i32, w: i32 },
    activation: extern struct { min: i32, max: i32 },
};

/// CMSIS-NN status codes shared across the accelerator bindings.
pub const ARM_CMSIS_NN_SUCCESS: i32 = 0;

/// Namespace exposing the CMSIS-NN entry points grouped by operator family.
/// Additional Helium kernels (e.g. pooling) can extend this pattern by adding
/// new structs next to `conv` while keeping the conditional stubs centralized.
pub const conv = if (is_enabled) struct {
    pub extern fn arm_convolve_wrapper_s8_get_buffer_size(
        conv_params: *const ConvParams,
        input_dims: *const Dims,
        filter_dims: *const Dims,
        output_dims: *const Dims,
    ) callconv(.C) i32;

    // Depthwise wrappers (s8)
    pub extern fn arm_depthwise_conv_wrapper_s8(
        ctx: *const Context,
        dw_conv_params: *const DwConvParams,
        quant_params: *const PerChannelQuantParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        filter_data: [*]const i8,
        bias_dims: *const Dims,
        bias_data: ?[*]const i32,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32;

    pub extern fn arm_depthwise_conv_wrapper_s8_get_buffer_size(
        dw_conv_params: *const DwConvParams,
        input_dims: *const Dims,
        filter_dims: *const Dims,
        output_dims: *const Dims,
    ) callconv(.C) i32;

    pub extern fn arm_convolve_wrapper_s8(
        ctx: *const Context,
        conv_params: *const ConvParams,
        quant_params: *const PerChannelQuantParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        filter_data: [*]const i8,
        bias_dims: *const Dims,
        bias_data: ?[*]const i32,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32;
    pub extern fn arm_convolve_s8_get_buffer_size(
        input_dims: *const Dims,
        filter_dims: *const Dims,
    ) callconv(.C) i32;

    pub extern fn arm_convolve_s8(
        ctx: *const Context,
        conv_params: *const ConvParams,
        quant_params: *const PerChannelQuantParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        filter_data: [*]const i8,
        bias_dims: *const Dims,
        bias_data: ?[*]const i32,
        upscale_dims: ?*const Dims,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32;
} else struct {
    pub fn arm_convolve_wrapper_s8_get_buffer_size(
        conv_params: *const ConvParams,
        input_dims: *const Dims,
        filter_dims: *const Dims,
        output_dims: *const Dims,
    ) callconv(.C) i32 {
        _ = conv_params;
        _ = input_dims;
        _ = filter_dims;
        _ = output_dims;
        return 0;
    }

    pub fn arm_convolve_wrapper_s8(
        ctx: *const Context,
        conv_params: *const ConvParams,
        quant_params: *const PerChannelQuantParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        filter_data: [*]const i8,
        bias_dims: *const Dims,
        bias_data: ?[*]const i32,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32 {
        _ = ctx;
        _ = conv_params;
        _ = quant_params;
        _ = input_dims;
        _ = input_data;
        _ = filter_dims;
        _ = filter_data;
        _ = bias_dims;
        _ = bias_data;
        _ = output_dims;
        _ = output_data;
        return 0;
    }

    pub fn arm_depthwise_conv_wrapper_s8(
        ctx: *const Context,
        dw_conv_params: *const DwConvParams,
        quant_params: *const PerChannelQuantParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        filter_data: [*]const i8,
        bias_dims: *const Dims,
        bias_data: ?[*]const i32,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32 {
        _ = ctx;
        _ = dw_conv_params;
        _ = quant_params;
        _ = input_dims;
        _ = input_data;
        _ = filter_dims;
        _ = filter_data;
        _ = bias_dims;
        _ = bias_data;
        _ = output_dims;
        _ = output_data;
        return 0;
    }

    pub fn arm_depthwise_conv_wrapper_s8_get_buffer_size(
        dw_conv_params: *const DwConvParams,
        input_dims: *const Dims,
        filter_dims: *const Dims,
        output_dims: *const Dims,
    ) callconv(.C) i32 {
        _ = dw_conv_params;
        _ = input_dims;
        _ = filter_dims;
        _ = output_dims;
        return 0;
    }
    pub fn arm_convolve_s8_get_buffer_size(
        input_dims: *const Dims,
        filter_dims: *const Dims,
    ) callconv(.C) i32 {
        _ = input_dims;
        _ = filter_dims;
        return 0;
    }

    pub fn arm_convolve_s8(
        ctx: *const Context,
        conv_params: *const ConvParams,
        quant_params: *const PerChannelQuantParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        filter_data: [*]const i8,
        bias_dims: *const Dims,
        bias_data: ?[*]const i32,
        upscale_dims: ?*const Dims,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32 {
        _ = ctx;
        _ = conv_params;
        _ = quant_params;
        _ = input_dims;
        _ = input_data;
        _ = filter_dims;
        _ = filter_data;
        _ = bias_dims;
        _ = bias_data;
        _ = upscale_dims;
        _ = output_dims;
        _ = output_data;
        return 0;
    }
};

pub inline fn isEnabled() bool {
    return is_enabled;
}

/// Helper queried by the TensorMath layer to understand if a Helium
/// implementation for QLinearConv can be attempted. Extend this (or add new
/// helpers) when wiring additional CMSIS-NN kernels such as MaxPool.
pub inline fn supportsConvolveS8() bool {
    return is_enabled;
}

/// Pooling parameter struct and s8 Average Pool externs
pub const PoolParams = extern struct {
    stride: extern struct { h: i32, w: i32 },
    padding: extern struct { h: i32, w: i32 },
    activation: extern struct { min: i32, max: i32 },
};

pub const pool = if (is_enabled) struct {
    pub extern fn arm_avgpool_s8(
        ctx: *const Context,
        pool_params: *const PoolParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32;

    pub extern fn arm_avgpool_s8_get_buffer_size(
        dim_dst_width: i32,
        ch_src: i32,
    ) callconv(.C) i32;
} else struct {
    pub fn arm_avgpool_s8(
        ctx: *const Context,
        pool_params: *const PoolParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32 {
        _ = ctx;
        _ = pool_params;
        _ = input_dims;
        _ = input_data;
        _ = filter_dims;
        _ = output_dims;
        _ = output_data;
        return 0;
    }

    pub fn arm_avgpool_s8_get_buffer_size(
        dim_dst_width: i32,
        ch_src: i32,
    ) callconv(.C) i32 {
        _ = dim_dst_width;
        _ = ch_src;
        return 0;
    }
};

pub inline fn supportsAvgPoolS8() bool {
    return is_enabled;
}
