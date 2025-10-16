const build_options = @import("build_options");

/// Indicates whether CMSIS Helium kernels are enabled for the current build.
const accel_requested = @hasDecl(build_options, "stm32n6_accel") and build_options.stm32n6_accel;
const cmsis_requested = @hasDecl(build_options, "stm32n6_use_cmsis") and build_options.stm32n6_use_cmsis;
const force_native = @hasDecl(build_options, "stm32n6_force_native") and build_options.stm32n6_force_native;
pub const is_enabled = accel_requested and cmsis_requested and !force_native;

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
    ) callconv(.c) i32;

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
    ) callconv(.c) i32;

    pub extern fn arm_depthwise_conv_wrapper_s8_get_buffer_size(
        dw_conv_params: *const DwConvParams,
        input_dims: *const Dims,
        filter_dims: *const Dims,
        output_dims: *const Dims,
    ) callconv(.c) i32;

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
    ) callconv(.c) i32;
    pub extern fn arm_convolve_s8_get_buffer_size(
        input_dims: *const Dims,
        filter_dims: *const Dims,
    ) callconv(.c) i32;

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
    ) callconv(.c) i32;
} else struct {
    pub fn arm_convolve_wrapper_s8_get_buffer_size(
        conv_params: *const ConvParams,
        input_dims: *const Dims,
        filter_dims: *const Dims,
        output_dims: *const Dims,
    ) callconv(.c) i32 {
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
    ) callconv(.c) i32 {
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
    ) callconv(.c) i32 {
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
    ) callconv(.c) i32 {
        _ = dw_conv_params;
        _ = input_dims;
        _ = filter_dims;
        _ = output_dims;
        return 0;
    }
    pub fn arm_convolve_s8_get_buffer_size(
        input_dims: *const Dims,
        filter_dims: *const Dims,
    ) callconv(.c) i32 {
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
    ) callconv(.c) i32 {
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
