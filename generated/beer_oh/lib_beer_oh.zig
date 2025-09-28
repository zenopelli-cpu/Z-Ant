const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const tensMath = zant.core.tensor.math_standard;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;
const utils = @import("codegen").codegen_v1.utils;
const param_lib = @import("static_parameters.zig");

// Global allocation tracking for safe deallocation
var last_result_size: usize = 0;

// Deallocator function for external C usage
pub export fn zant_free_result(ptr: ?[*]T_out) callconv(.C) void {
    if (ptr) |valid_ptr| {
        if (last_result_size > 0) {
            const slice = valid_ptr[0..last_result_size];
            allocator.free(slice);
            last_result_size = 0;
        }
    }
}

var log_function: ?*const fn ([*c]const u8) callconv(.C) void = null;
var log_buffer: [512]u8 = undefined;

fn safe_log_forwarder(c_msg: [*c]const u8) callconv(.C) void {
    if (log_function) |user_log| {
        var i: usize = 0;
        while (i < log_buffer.len - 1 and c_msg[i] != 0) : (i += 1) {
            log_buffer[i] = c_msg[i];
        }
        log_buffer[i] = 0;
        user_log(@ptrCast(&log_buffer));
    }
}

pub export fn setLogFunction(func: ?*const fn ([*c]const u8) callconv(.C) void) void {
    // Keep only our own logging; disable verbose internal logs
    log_function = func;
    zant.core.tensor.setLogFunction(null);
    tensMath.setQLinearConvLogFunctionC(null);
}

const T_in: type = f32;
const T_out: type = f32;
inline fn logMsg(comptime msg: []const u8) void {
    if (log_function != null) {
        var buffer: [msg.len + 1:0]u8 = undefined;
        @memcpy(buffer[0..msg.len], msg);
        buffer[msg.len] = 0;
        safe_log_forwarder(@ptrCast(&buffer));
    }
}

inline fn logf(comptime fmt: []const u8, args: anytype) void {
    if (log_function != null) {
        var tmp: [256]u8 = undefined;
        const written = std.fmt.bufPrint(&tmp, fmt, args) catch return;
        var zbuf: [257:0]u8 = undefined;
        @memcpy(zbuf[0..written.len], written);
        zbuf[written.len] = 0;
        safe_log_forwarder(@ptrCast(&zbuf));
    }
}

fn logTensorStatsU8(label: []const u8, t: *Tensor(u8)) void {
    var min_val: u8 = 255;
    var max_val: u8 = 0;
    var sum_val: usize = 0;
    var zeros: usize = 0;
    var i: usize = 0;
    while (i < t.size) : (i += 1) {
        const v = t.data[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum_val += v;
        if (v == 0) zeros += 1;
    }
    const mean_val: f32 = @as(f32, @floatFromInt(sum_val)) / @as(f32, @floatFromInt(t.size));
    const s0 = if (t.shape.len > 0) t.shape[0] else 0;
    const s1 = if (t.shape.len > 1) t.shape[1] else 0;
    const s2 = if (t.shape.len > 2) t.shape[2] else 0;
    const s3 = if (t.shape.len > 3) t.shape[3] else 0;
    logf("[conv] {s} shape={{ {}, {}, {}, {} }} size={} min={} max={} mean={} zeros={}\\n", .{ label, s0, s1, s2, s3, t.size, min_val, max_val, mean_val, zeros });
}

// return codes:
//  0 : everything good
// -1 : something went wrong in the mathematical operations
// -2 : something went wrong in the initialization phase
// -3 : something went wrong in the output/return phase
pub export fn predict(
    input: [*]T_in,
    input_shape: [*]u32,
    shape_len: u32,
    result: *[*]T_out,
) callconv(.C) i32 {
    logMsg("Starting prediction...\\n");
    //checks on the input parameters
    if (shape_len == 0) return -2;
    if (shape_len != 4) return -2;
    if (input_shape[0] != 1) return -2;
    if (input_shape[1] != 96) return -2;
    if (input_shape[2] != 96) return -2;
    if (input_shape[3] != 1) return -2;
    //computing the size of the input tensor (runtime)
    var input_size: usize = 1;
    for (0..shape_len) |dim_i| {
        input_size *= @as(usize, input_shape[dim_i]);
    }
    // Fixed input shape (validated above)
    var input_shape_fixed: [4]usize = .{ 1, 96, 96, 1 };

    // Zero-copy tensor pointing directly to input data
    var tensor_serving_default_x_0 = Tensor(T_in){
        .data = input[0..input_size],
        .shape = input_shape_fixed[0..],
        .size = input_size,
        .allocator = &allocator, // non-owning view
    };
    logMsg("Using plan-based execution with 37 steps\\n");

    var shape_tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0: [4]usize = [_]usize{ 1, 1, 96, 96 };
    var tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0) catch return -2;
    defer tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0.deinit();

    // Step 0: reshape operation

    logMsg("Running reshape operation...\\n");

    // Reshape Operation
    // Convert shape tensor data to isize slice
    // Pass the local allocator to the utils function
    const shape_slice_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0 = utils.sliceToIsizeSlice(allocator, param_lib.tensor_new_shape__154.data); // Removed catch return
    defer allocator.free(shape_slice_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0); // Free the runtime allocated slice // Generated shape slice code

    tensMath.reshape_lean(
        f32, // Use actual input tensor type
        @constCast(&tensor_serving_default_x_0),
        shape_slice_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0, // Pre-built shape slice argument
        false, // Format boolean correctly
        &tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0, // Pre-built output tensor argument
    ) catch return -1;

    var shape_tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_quantized: [4]usize = [_]usize{ 1, 1, 96, 96 };
    var tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_quantized) catch return -2;
    defer tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_quantized.deinit();

    // Step 1: quantizelinear operation

    logMsg("Running quantizelinear operation...\\n");

    tensMath.quantizeLinear_lean(
        f32, // InputType
        u8, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0, // x: input tensor
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_quantized, // y: output tensor
    ) catch return -1;
    tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0.deinit();

    var shape_tensor_relu6__5_0_quantized: [4]usize = [_]usize{ 1, 16, 48, 48 };
    var tensor_relu6__5_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__5_0_quantized) catch return -2;
    defer tensor_relu6__5_0_quantized.deinit();

    // Step 2: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_quantized), // input x
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__159_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__159_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__159_zero_point), // w_zero_point
        &tensor_relu6__5_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_bn_conv1_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 2, 2 }, // stride
        &[_]usize{ 1, 1, 1, 1 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__5_0_quantized);
    }
    tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_quantized.deinit();

    var shape_tensor_relu6__7_0_quantized: [4]usize = [_]usize{ 1, 16, 48, 48 };
    var tensor_relu6__7_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__7_0_quantized) catch return -2;
    defer tensor_relu6__7_0_quantized.deinit();

    // Step 3: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__5_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__190_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__190_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__159_zero_point), // w_zero_point
        &tensor_relu6__7_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 1, 1, 1, 1 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        16, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__7_0_quantized);
    }
    tensor_relu6__5_0_quantized.deinit();

    var shape_tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized: [4]usize = [_]usize{ 1, 8, 48, 48 };
    var tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized) catch return -2;
    defer tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized.deinit();

    // Step 4: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__7_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__193_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__193_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__193_zero_point), // w_zero_point
        &tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized, // output
        @constCast(&param_lib.tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized);
    }
    tensor_relu6__7_0_quantized.deinit();

    var shape_tensor_relu6__10_0_quantized: [4]usize = [_]usize{ 1, 48, 48, 48 };
    var tensor_relu6__10_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__10_0_quantized) catch return -2;
    defer tensor_relu6__10_0_quantized.deinit();

    // Step 5: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized), // input x
        @constCast(&param_lib.tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__181_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__181_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__181_zero_point), // w_zero_point
        &tensor_relu6__10_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_1_expand_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__10_0_quantized);
    }
    tensor_model_1_expanded_conv_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_expanded_conv_project_conv2d1_quantized.deinit();

    var shape_tensor_relu6__12_0_quantized: [4]usize = [_]usize{ 1, 48, 24, 24 };
    var tensor_relu6__12_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__12_0_quantized) catch return -2;
    defer tensor_relu6__12_0_quantized.deinit();

    // Step 6: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__10_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__187_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__187_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__181_zero_point), // w_zero_point
        &tensor_relu6__12_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_1_depthwise_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 2, 2 }, // stride
        &[_]usize{ 0, 0, 1, 1 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        48, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__12_0_quantized);
    }
    tensor_relu6__10_0_quantized.deinit();

    var shape_tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized: [4]usize = [_]usize{ 1, 8, 24, 24 };
    var tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized) catch return -2;
    defer tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized.deinit();

    // Step 7: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__12_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__176_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__176_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__193_zero_point), // w_zero_point
        &tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized, // output
        @constCast(&param_lib.tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_block_1_depthwise_relu_relu6_model_1_block_1_depthwise_bn_fusedbatchnormv3_model_1_block_3_depthwise_bn_fusedbatchnormv3_model_1_block_3_depthwise_depthwise_model_1_block_1_depthwise_depthwise_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized);
    }
    tensor_relu6__12_0_quantized.deinit();

    var shape_tensor_relu6__15_0_quantized: [4]usize = [_]usize{ 1, 48, 24, 24 };
    var tensor_relu6__15_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__15_0_quantized) catch return -2;
    defer tensor_relu6__15_0_quantized.deinit();

    // Step 8: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized), // input x
        @constCast(&param_lib.tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_1_depthwise_relu_relu6_model_1_block_1_depthwise_bn_fusedbatchnormv3_model_1_block_3_depthwise_bn_fusedbatchnormv3_model_1_block_3_depthwise_depthwise_model_1_block_1_depthwise_depthwise_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__173_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__173_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__181_zero_point), // w_zero_point
        &tensor_relu6__15_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_2_expand_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__15_0_quantized);
    }

    var shape_tensor_relu6__17_0_quantized: [4]usize = [_]usize{ 1, 48, 24, 24 };
    var tensor_relu6__17_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__17_0_quantized) catch return -2;
    defer tensor_relu6__17_0_quantized.deinit();

    // Step 9: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__15_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__184_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__184_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__181_zero_point), // w_zero_point
        &tensor_relu6__17_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_2_depthwise_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 1, 1, 1, 1 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        48, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__17_0_quantized);
    }
    tensor_relu6__15_0_quantized.deinit();

    var shape_tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized: [4]usize = [_]usize{ 1, 8, 24, 24 };
    var tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized) catch return -2;
    defer tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized.deinit();

    // Step 10: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__17_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__169_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__169_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__193_zero_point), // w_zero_point
        &tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized, // output
        @constCast(&param_lib.tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized);
    }
    tensor_relu6__17_0_quantized.deinit();

    var shape_tensor_model_1_block_2_add_add_quantized: [4]usize = [_]usize{ 1, 8, 24, 24 };
    var tensor_model_1_block_2_add_add_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_2_add_add_quantized) catch return -2;
    defer tensor_model_1_block_2_add_add_quantized.deinit();

    // Step 11: fused_dequantizelinear_dequantizelinear_add_quantizelinear operation

    logMsg("Running fused_dequantizelinear_dequantizelinear_add_quantizelinear operation...\\n");

    tensMath.qlinearadd_lean(
        @constCast(&tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized),
        @constCast(&param_lib.tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_scale),
        @constCast(&param_lib.tensor_model_1_block_1_depthwise_relu_relu6_model_1_block_1_depthwise_bn_fusedbatchnormv3_model_1_block_3_depthwise_bn_fusedbatchnormv3_model_1_block_3_depthwise_depthwise_model_1_block_1_depthwise_depthwise_zero_point),
        @constCast(&tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized),
        @constCast(&param_lib.tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_scale),
        @constCast(&param_lib.tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_zero_point),
        &tensor_model_1_block_2_add_add_quantized,
        @constCast(&param_lib.tensor_model_1_block_2_add_add_scale),
        @constCast(&param_lib.tensor_model_1_block_2_add_add_zero_point),
    ) catch return -1;
    tensor_model_1_block_2_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d1_quantized.deinit();
    tensor_model_1_block_1_project_bn_fusedbatchnormv3_model_1_block_2_project_conv2d_model_1_block_1_project_conv2d1_quantized.deinit();

    var shape_tensor_relu6__20_0_quantized: [4]usize = [_]usize{ 1, 48, 24, 24 };
    var tensor_relu6__20_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__20_0_quantized) catch return -2;
    defer tensor_relu6__20_0_quantized.deinit();

    // Step 12: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_model_1_block_2_add_add_quantized), // input x
        @constCast(&param_lib.tensor_model_1_block_2_add_add_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_2_add_add_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__162_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__162_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__181_zero_point), // w_zero_point
        &tensor_relu6__20_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_3_expand_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__20_0_quantized);
    }
    tensor_model_1_block_2_add_add_quantized.deinit();

    var shape_tensor_relu6__22_0_quantized: [4]usize = [_]usize{ 1, 48, 12, 12 };
    var tensor_relu6__22_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__22_0_quantized) catch return -2;
    defer tensor_relu6__22_0_quantized.deinit();

    // Step 13: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__20_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__180_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__180_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__181_zero_point), // w_zero_point
        &tensor_relu6__22_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_3_depthwise_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 2, 2 }, // stride
        &[_]usize{ 0, 0, 1, 1 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        48, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__22_0_quantized);
    }
    tensor_relu6__20_0_quantized.deinit();

    var shape_tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized) catch return -2;
    defer tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized.deinit();

    // Step 14: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__22_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__189_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__189_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__159_zero_point), // w_zero_point
        &tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized, // output
        @constCast(&param_lib.tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized);
    }
    tensor_relu6__22_0_quantized.deinit();

    var shape_tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1 = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1) catch return -2;
    defer tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1.deinit();

    // Step 15: dequantizelinear operation

    logMsg("Running dequantizelinear operation...\\n");

    tensMath.dequantizeLinear_lean(
        u8, // InputType
        f32, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized, // x: input tensor
        @constCast(&param_lib.tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_zero_point), // x_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1, // y: output tensor
    ) catch return -1;

    var shape_tensor_relu6__25_0_quantized: [4]usize = [_]usize{ 1, 96, 12, 12 };
    var tensor_relu6__25_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__25_0_quantized) catch return -2;
    defer tensor_relu6__25_0_quantized.deinit();

    // Step 16: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized), // input x
        @constCast(&param_lib.tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__203_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__203_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__203_zero_point), // w_zero_point
        &tensor_relu6__25_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_4_expand_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__25_0_quantized);
    }
    tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1_quantized.deinit();

    var shape_tensor_relu6__27_0_quantized: [4]usize = [_]usize{ 1, 96, 12, 12 };
    var tensor_relu6__27_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__27_0_quantized) catch return -2;
    defer tensor_relu6__27_0_quantized.deinit();

    // Step 17: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__25_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__164_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__164_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__203_zero_point), // w_zero_point
        &tensor_relu6__27_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_4_depthwise_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 1, 1, 1, 1 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        96, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__27_0_quantized);
    }
    tensor_relu6__25_0_quantized.deinit();

    var shape_tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized) catch return -2;
    defer tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized.deinit();

    // Step 18: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__27_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__198_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__198_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__159_zero_point), // w_zero_point
        &tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized, // output
        @constCast(&param_lib.tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized);
    }
    tensor_relu6__27_0_quantized.deinit();

    var shape_tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1 = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1) catch return -2;
    defer tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1.deinit();

    // Step 19: dequantizelinear operation

    logMsg("Running dequantizelinear operation...\\n");

    tensMath.dequantizeLinear_lean(
        u8, // InputType
        f32, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized, // x: input tensor
        @constCast(&param_lib.tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_zero_point), // x_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1, // y: output tensor
    ) catch return -1;
    tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1_quantized.deinit();

    var shape_tensor_model_1_block_4_add_add: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_4_add_add = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_block_4_add_add) catch return -2;
    defer tensor_model_1_block_4_add_add.deinit();

    // Step 20: add operation

    logMsg("Running add operation...\\n");

    tensMath.sum_tensors_lean(f32, f32, &tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1, &tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1, &tensor_model_1_block_4_add_add) catch return -1;
    tensor_model_1_block_3_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_3_project_conv2d1.deinit();
    tensor_model_1_block_4_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_model_1_block_4_project_conv2d1.deinit();

    var shape_tensor_model_1_block_4_add_add_quantized: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_4_add_add_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_4_add_add_quantized) catch return -2;
    defer tensor_model_1_block_4_add_add_quantized.deinit();

    // Step 21: quantizelinear operation

    logMsg("Running quantizelinear operation...\\n");

    tensMath.quantizeLinear_lean(
        f32, // InputType
        u8, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_block_4_add_add, // x: input tensor
        @constCast(&param_lib.tensor_model_1_block_4_add_add_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_block_4_add_add_zero_point), // y_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_block_4_add_add_quantized, // y: output tensor
    ) catch return -1;

    var shape_tensor_relu6__30_0_quantized: [4]usize = [_]usize{ 1, 96, 12, 12 };
    var tensor_relu6__30_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__30_0_quantized) catch return -2;
    defer tensor_relu6__30_0_quantized.deinit();

    // Step 22: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_model_1_block_4_add_add_quantized), // input x
        @constCast(&param_lib.tensor_model_1_block_4_add_add_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_4_add_add_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__197_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__197_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__203_zero_point), // w_zero_point
        &tensor_relu6__30_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__30_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_5_expand_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__30_0_quantized);
    }
    tensor_model_1_block_4_add_add_quantized.deinit();

    var shape_tensor_relu6__32_0_quantized: [4]usize = [_]usize{ 1, 96, 12, 12 };
    var tensor_relu6__32_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__32_0_quantized) catch return -2;
    defer tensor_relu6__32_0_quantized.deinit();

    // Step 23: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__30_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__30_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__201_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__201_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__203_zero_point), // w_zero_point
        &tensor_relu6__32_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__5_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_5_depthwise_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 1, 1, 1, 1 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        96, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__32_0_quantized);
    }
    tensor_relu6__30_0_quantized.deinit();

    var shape_tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized) catch return -2;
    defer tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized.deinit();

    // Step 24: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__32_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__5_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__202_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__202_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__159_zero_point), // w_zero_point
        &tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized, // output
        @constCast(&param_lib.tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized);
    }
    tensor_relu6__32_0_quantized.deinit();

    var shape_tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1 = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1) catch return -2;
    defer tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1.deinit();

    // Step 25: dequantizelinear operation

    logMsg("Running dequantizelinear operation...\\n");

    tensMath.dequantizeLinear_lean(
        u8, // InputType
        f32, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized, // x: input tensor
        @constCast(&param_lib.tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_zero_point), // x_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1, // y: output tensor
    ) catch return -1;
    tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1_quantized.deinit();

    var shape_tensor_model_1_block_5_add_add: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_5_add_add = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_block_5_add_add) catch return -2;
    defer tensor_model_1_block_5_add_add.deinit();

    // Step 26: add operation

    logMsg("Running add operation...\\n");

    tensMath.sum_tensors_lean(f32, f32, &tensor_model_1_block_4_add_add, &tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1, &tensor_model_1_block_5_add_add) catch return -1;
    tensor_model_1_block_5_project_bn_fusedbatchnormv3_model_1_block_5_project_conv2d1.deinit();
    tensor_model_1_block_4_add_add.deinit();

    var shape_tensor_model_1_block_5_add_add_quantized: [4]usize = [_]usize{ 1, 16, 12, 12 };
    var tensor_model_1_block_5_add_add_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_block_5_add_add_quantized) catch return -2;
    defer tensor_model_1_block_5_add_add_quantized.deinit();

    // Step 27: quantizelinear operation

    logMsg("Running quantizelinear operation...\\n");

    tensMath.quantizeLinear_lean(
        f32, // InputType
        u8, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_block_5_add_add, // x: input tensor
        @constCast(&param_lib.tensor_model_1_block_5_add_add_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_block_4_add_add_zero_point), // y_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_block_5_add_add_quantized, // y: output tensor
    ) catch return -1;
    tensor_model_1_block_5_add_add.deinit();

    var shape_tensor_relu6__35_0_quantized: [4]usize = [_]usize{ 1, 96, 12, 12 };
    var tensor_relu6__35_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu6__35_0_quantized) catch return -2;
    defer tensor_relu6__35_0_quantized.deinit();

    // Step 28: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_model_1_block_5_add_add_quantized), // input x
        @constCast(&param_lib.tensor_model_1_block_5_add_add_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_block_4_add_add_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__199_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__199_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__203_zero_point), // w_zero_point
        &tensor_relu6__35_0_quantized, // output
        @constCast(&param_lib.tensor_relu6__35_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_block_6_expand_bn_fusedbatchnormv3_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_relu6__35_0_quantized);
    }
    tensor_model_1_block_5_add_add_quantized.deinit();

    var shape_tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized: [4]usize = [_]usize{ 1, 32, 12, 12 };
    var tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized) catch return -2;
    defer tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized.deinit();

    // Step 29: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu6__35_0_quantized), // input x
        @constCast(&param_lib.tensor_relu6__35_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__194_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__194_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__194_zero_point), // w_zero_point
        &tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized, // output
        @constCast(&param_lib.tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_head_bias_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0, 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized);
    }
    tensor_relu6__35_0_quantized.deinit();

    var shape_tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias: [4]usize = [_]usize{ 1, 32, 12, 12 };
    var tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias) catch return -2;
    defer tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias.deinit();

    // Step 30: dequantizelinear operation

    logMsg("Running dequantizelinear operation...\\n");

    tensMath.dequantizeLinear_lean(
        u8, // InputType
        f32, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized, // x: input tensor
        @constCast(&param_lib.tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_zero_point), // x_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias, // y: output tensor
    ) catch return -1;
    tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias_quantized.deinit();

    var shape_tensor_relu__37_0: [4]usize = [_]usize{ 1, 32, 12, 12 };
    var tensor_relu__37_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_relu__37_0) catch return -2;
    defer tensor_relu__37_0.deinit();

    // Step 31: relu operation

    logMsg("Running relu operation...\\n");

    tensMath.ReLU_lean(f32, &tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias, &tensor_relu__37_0) catch return -1;
    tensor_model_1_head_relu_model_1_head_biasadd_model_1_head_conv2d_head_bias.deinit();

    var shape_tensor_relu__37_0_quantized: [4]usize = [_]usize{ 1, 32, 12, 12 };
    var tensor_relu__37_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_relu__37_0_quantized) catch return -2;
    defer tensor_relu__37_0_quantized.deinit();

    // Step 32: quantizelinear operation

    logMsg("Running quantizelinear operation...\\n");

    tensMath.quantizeLinear_lean(
        f32, // InputType
        u8, // OutputType
        u8, // ZeroPointType
        &tensor_relu__37_0, // x: input tensor
        @constCast(&param_lib.tensor_relu__37_0_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // y_zero_point
        1, // axis
        0, // block_size
        &tensor_relu__37_0_quantized, // y: output tensor
    ) catch return -1;
    tensor_relu__37_0.deinit();

    var shape_tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized: [4]usize = [_]usize{ 1, 3, 12, 12 };
    var tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized) catch return -2;
    defer tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized.deinit();

    // Step 33: qlinearconv operation

    logMsg("Running qlinearconv operation...\\n");
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_relu__37_0_quantized), // input x
        @constCast(&param_lib.tensor_relu__37_0_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_conv1_relu_relu6_model_1_bn_conv1_fusedbatchnormv3_model_1_expanded_conv_depthwise_bn_fusedbatchnormv3_model_1_expanded_conv_depthwise_depthwise_model_1_block_5_project_conv2d_model_1_conv1_conv2d__40_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_const_fold_opt__171_quantized), // w
        @constCast(&param_lib.tensor_const_fold_opt__171_scale), // w_scale
        @constCast(&param_lib.tensor_const_fold_opt__171_zero_point), // w_zero_point
        &tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized, // output
        @constCast(&param_lib.tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_scale), // y_scale
        @constCast(&param_lib.tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias_quantized), // bias
        &[_]usize{ 1, 1 }, // stride
        &[_]usize{ 0, 0, 0, 0, 0, 0, 0, 0 }, // pads
        &[_]usize{ 1, 1 }, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;
    if (log_function != null) {
        logTensorStatsU8("qconv", &tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized);
    }
    tensor_relu__37_0_quantized.deinit();

    var shape_tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1: [4]usize = [_]usize{ 1, 3, 12, 12 };
    var tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1 = Tensor(f32).fromShape(&allocator, &shape_tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1) catch return -2;
    defer tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1.deinit();

    // Step 34: dequantizelinear operation

    logMsg("Running dequantizelinear operation...\\n");

    tensMath.dequantizeLinear_lean(
        u8, // InputType
        f32, // OutputType
        u8, // ZeroPointType
        &tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized, // x: input tensor
        @constCast(&param_lib.tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_scale), // x_scale
        @constCast(&param_lib.tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_zero_point), // x_zero_point
        1, // axis
        0, // block_size
        &tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1, // y: output tensor
    ) catch return -1;
    tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1_quantized.deinit();

    var shape_tensor_statefulpartitionedcall_0_raw_output___3_0: [4]usize = [_]usize{ 1, 3, 12, 12 };
    var tensor_statefulpartitionedcall_0_raw_output___3_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_statefulpartitionedcall_0_raw_output___3_0) catch return -2;
    defer tensor_statefulpartitionedcall_0_raw_output___3_0.deinit();

    // Step 35: softmax operation

    logMsg("Running softmax operation...\\n");

    tensMath.softmax_lean(
        f32, //Type
        &tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1, // input tensor
        &tensor_statefulpartitionedcall_0_raw_output___3_0, // output tensor
        1,
    ) catch return -1;
    tensor_model_1_logits_biasadd_model_1_logits_conv2d_logits_bias1.deinit();

    var shape_tensor_statefulpartitionedcall_0: [4]usize = [_]usize{ 1, 12, 12, 3 };
    var tensor_statefulpartitionedcall_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_statefulpartitionedcall_0) catch return -2;

    // Step 36: transpose operation

    logMsg("Running transpose operation...\\n");
    tensMath.transpose_onnx_lean(
        f32, //input type
        &tensor_statefulpartitionedcall_0_raw_output___3_0, // input tensor
        &[_]usize{ 0, 2, 3, 1 }, // perm array
        &tensor_statefulpartitionedcall_0, // output
        allocator,
    ) catch return -1;
    tensor_statefulpartitionedcall_0_raw_output___3_0.deinit();

    const output_zant_slice = allocator.alloc(T_out, tensor_statefulpartitionedcall_0.size) catch return -3;
    @memcpy(output_zant_slice, tensor_statefulpartitionedcall_0.data[0..tensor_statefulpartitionedcall_0.size]);

    // Track allocation so zant_free_result can release it later
    last_result_size = tensor_statefulpartitionedcall_0.size;

    // Deallocate the output tensor after copying its data
    tensor_statefulpartitionedcall_0.deinit();

    //The Caller must handle the memory of output_zant_slice
    result.* = output_zant_slice.ptr;

    logMsg("Prediction completed.\\n");
    return 0;
}
