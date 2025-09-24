
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
pub  fn zant_free_result(ptr: ?[*]T_out) callconv(.C) void {
    if (ptr) |valid_ptr| {
        if (last_result_size > 0) {
            const slice = valid_ptr[0..last_result_size];
            allocator.free(slice);
            last_result_size = 0;
        }
    }
}

 const T_in : type = f32;
 const T_out : type = f32;
 // return codes:
 //  0 : everything good
 // -1 : something went wrong in the mathematical operations
 // -2 : something went wrong in the initialization phase
 // -3 : something went wrong in the output/return phase
pub  fn predict (
    input: [*]T_in,
    input_shape: [*]u32,
    shape_len: u32,
    result: *[*]T_out,
)  i32 { 
    //checks on the input parameters
    if (shape_len == 0) return -2;
    if(shape_len != 4) return -2;
    if( input_shape[0] != 1) return -2;
    if( input_shape[1] != 3) return -2;
    if( input_shape[2] != 96) return -2;
    if( input_shape[3] != 96) return -2;  
    //computing the size of the input tensor (runtime)
    var input_size: usize = 1;
    for(0..shape_len) |dim_i| {
        input_size *= @as(usize, input_shape[dim_i]);
    }
    // Build runtime input shape from caller (u32 -> usize)
    var input_shape_runtime = allocator.alloc(usize, shape_len) catch return -2;
    defer allocator.free(input_shape_runtime);
    for (0..shape_len) |i| {
        input_shape_runtime[i] = @as(usize, input_shape[i]);
    }

    // Zero-copy tensor pointing directly to input data
    var tensor_images = Tensor(T_in){
        .data = input[0..input_size],
        .shape = input_shape_runtime[0..],
        .size = input_size,
        .allocator = &allocator, // non-owning view
    };

var shape_tensor_images_quantized : [4]usize = [_]usize{ 1, 3, 96, 96} ;
    var tensor_images_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor_images_quantized) catch return -2;
    defer tensor_images_quantized.deinit();

   // Step 0: quantizelinear operation


    tensMath.quantizeLinear_lean(f32, // InputType
                                 u8, // OutputType
                                 u8, // ZeroPointType
                                 &tensor_images, // x: input tensor
                                 @constCast(&param_lib.tensor_images_scale), // y_scale
                                 @constCast(&param_lib.tensor_images_zero_point), // y_zero_point
                                 1,  // axis
                                 0,  // block_size
                                 &tensor_images_quantized, // y: output tensor
    ) catch return -1;

var shape_tensor__model_backbone_features_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 48, 48} ;
    var tensor__model_backbone_features_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_0_conv_conv_output_0_quantized.deinit();

   // Step 1: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor_images_quantized), // input x
        @constCast(&param_lib.tensor_images_scale), // x_scale
        @constCast(&param_lib.tensor_images_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_497_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_497_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_498_quantized), // bias
        &[_]usize{2,2}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor_images_quantized.deinit();


var shape_tensor__model_backbone_features_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 24, 24} ;
    var tensor__model_backbone_features_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_1_conv_conv_output_0_quantized.deinit();

   // Step 2: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_500_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_500_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_501_quantized), // bias
        &[_]usize{2,2}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_0_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 24, 24} ;
    var tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 3: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_503_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_503_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_504_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_1_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_2_skip_averagepool_output_0_quantized : [4]usize = [_]usize{ 1, 128, 12, 12} ;
    var tensor__model_backbone_features_2_skip_averagepool_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_2_skip_averagepool_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_2_skip_averagepool_output_0_quantized.deinit();

   // Step 4: qlinearaveragepool operation

    // QLinearAveragePool attributes
    var kernel_shape__model_backbone_features_2_skip_averagepool_output_0_quantized = [_]usize{3, 3};
    var strides__model_backbone_features_2_skip_averagepool_output_0_quantized = [_]usize{2, 2};
    var dilations__model_backbone_features_2_skip_averagepool_output_0_quantized = [_]usize{1, 1};
    var pads__model_backbone_features_2_skip_averagepool_output_0_quantized = [_]usize{1, 1, 1, 1};
    const auto_pad__model_backbone_features_2_skip_averagepool_output_0_quantized = tensMath.AutoPadType.NOTSET;

    // Perform QLinearAveragePool
    tensMath.lean_qlinearaveragepool(
        u8,
        f32,
        u8,
        @constCast(&tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized),
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_scale),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        @constCast(&param_lib.tensor__model_backbone_features_2_skip_averagepool_output_0_scale),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        &tensor__model_backbone_features_2_skip_averagepool_output_0_quantized,
        &kernel_shape__model_backbone_features_2_skip_averagepool_output_0_quantized,
        &strides__model_backbone_features_2_skip_averagepool_output_0_quantized,
        &dilations__model_backbone_features_2_skip_averagepool_output_0_quantized,
        &pads__model_backbone_features_2_skip_averagepool_output_0_quantized,
        auto_pad__model_backbone_features_2_skip_averagepool_output_0_quantized,
        true,
    ) catch return -1;


var shape_tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 12, 12} ;
    var tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_quantized.deinit();

   // Step 5: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_506_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_506_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_507_quantized), // bias
        &[_]usize{2,2}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        128, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_2_conv_list_0_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 12, 12} ;
    var tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 6: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_509_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_509_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_510_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_2_avd_layer_avd_layer_0_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 7: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_512_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_512_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_513_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 8: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_515_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_515_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_516_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_2_concat_output_0_quantized : [4]usize = [_]usize{ 1, 256, 12, 12} ;
    var tensor__model_backbone_features_2_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_2_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_2_concat_output_0_quantized.deinit();

   // Step 9: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_2_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_2_skip_averagepool_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_2_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_2_skip_averagepool_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_2_skip_averagepool_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_2_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_2_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_2_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_2_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_2_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_2_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_2_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_2_skip_averagepool_output_0_quantized.deinit();
    tensor__model_backbone_features_2_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_2_conv_list_3_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 12, 12} ;
    var tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 10: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_2_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_2_skip_averagepool_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_518_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_518_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_519_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_2_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 12, 12} ;
    var tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 11: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_521_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_521_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_522_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 12: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_524_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_524_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_525_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 13: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_527_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_527_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_528_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_3_concat_output_0_quantized : [4]usize = [_]usize{ 1, 256, 12, 12} ;
    var tensor__model_backbone_features_3_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_3_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_3_concat_output_0_quantized.deinit();

   // Step 14: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_3_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_3_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_3_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_3_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_3_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_3_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_3_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_3_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_3_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_3_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_3_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 12, 12} ;
    var tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 15: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_3_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_3_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_530_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_530_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_531_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_3_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 12, 12} ;
    var tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 16: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_533_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_533_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_534_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 17: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_536_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_536_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_537_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 18: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_539_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_539_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_540_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_4_concat_output_0_quantized : [4]usize = [_]usize{ 1, 256, 12, 12} ;
    var tensor__model_backbone_features_4_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_4_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_4_concat_output_0_quantized.deinit();

   // Step 19: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_4_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_4_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_4_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_4_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_4_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_4_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_4_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_4_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_4_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_4_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_4_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 12, 12} ;
    var tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 20: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_4_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_4_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_542_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_542_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_543_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_4_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 12, 12} ;
    var tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 21: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_545_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_545_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_546_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 22: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_548_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_548_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_549_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 32, 12, 12} ;
    var tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 23: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_551_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_551_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_552_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_5_concat_output_0_quantized : [4]usize = [_]usize{ 1, 256, 12, 12} ;
    var tensor__model_backbone_features_5_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_5_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_5_concat_output_0_quantized.deinit();

   // Step 24: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_5_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_5_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_5_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_5_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_5_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_5_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_5_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_5_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_5_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_5_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_5_conv_list_0_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 12, 12} ;
    var tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 25: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_5_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_5_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_554_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_554_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_555_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_5_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_6_skip_averagepool_output_0_quantized : [4]usize = [_]usize{ 1, 256, 6, 6} ;
    var tensor__model_backbone_features_6_skip_averagepool_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_6_skip_averagepool_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_6_skip_averagepool_output_0_quantized.deinit();

   // Step 26: qlinearaveragepool operation

    // QLinearAveragePool attributes
    var kernel_shape__model_backbone_features_6_skip_averagepool_output_0_quantized = [_]usize{3, 3};
    var strides__model_backbone_features_6_skip_averagepool_output_0_quantized = [_]usize{2, 2};
    var dilations__model_backbone_features_6_skip_averagepool_output_0_quantized = [_]usize{1, 1};
    var pads__model_backbone_features_6_skip_averagepool_output_0_quantized = [_]usize{1, 1, 1, 1};
    const auto_pad__model_backbone_features_6_skip_averagepool_output_0_quantized = tensMath.AutoPadType.NOTSET;

    // Perform QLinearAveragePool
    tensMath.lean_qlinearaveragepool(
        u8,
        f32,
        u8,
        @constCast(&tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized),
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_scale),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        @constCast(&param_lib.tensor__model_backbone_features_6_skip_averagepool_output_0_scale),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        &tensor__model_backbone_features_6_skip_averagepool_output_0_quantized,
        &kernel_shape__model_backbone_features_6_skip_averagepool_output_0_quantized,
        &strides__model_backbone_features_6_skip_averagepool_output_0_quantized,
        &dilations__model_backbone_features_6_skip_averagepool_output_0_quantized,
        &pads__model_backbone_features_6_skip_averagepool_output_0_quantized,
        auto_pad__model_backbone_features_6_skip_averagepool_output_0_quantized,
        true,
    ) catch return -1;


var shape_tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 6, 6} ;
    var tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_quantized.deinit();

   // Step 27: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_557_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_557_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_558_quantized), // bias
        &[_]usize{2,2}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        256, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_6_conv_list_0_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 6, 6} ;
    var tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 28: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_560_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_560_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_561_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_6_avd_layer_avd_layer_0_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 29: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_563_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_563_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_564_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 30: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_566_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_566_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_567_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_6_concat_output_0_quantized : [4]usize = [_]usize{ 1, 512, 6, 6} ;
    var tensor__model_backbone_features_6_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_6_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_6_concat_output_0_quantized.deinit();

   // Step 31: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_6_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_6_skip_averagepool_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_6_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_6_skip_averagepool_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_6_skip_averagepool_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_6_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_6_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_6_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_6_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_6_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_6_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_6_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_6_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_6_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_6_skip_averagepool_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 6, 6} ;
    var tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 32: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_6_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_6_skip_averagepool_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_569_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_569_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_570_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_6_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 6, 6} ;
    var tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 33: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_572_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_572_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_573_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 34: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_575_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_575_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_576_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 35: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_578_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_578_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_579_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_7_concat_output_0_quantized : [4]usize = [_]usize{ 1, 512, 6, 6} ;
    var tensor__model_backbone_features_7_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_7_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_7_concat_output_0_quantized.deinit();

   // Step 36: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_7_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_7_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_7_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_7_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_7_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_7_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_7_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_7_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_7_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_7_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_7_conv_list_1_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 6, 6} ;
    var tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 37: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_7_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_7_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_581_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_581_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_582_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_7_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 6, 6} ;
    var tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 38: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_584_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_584_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_585_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 39: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_587_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_587_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_588_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 40: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_590_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_590_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_591_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_8_concat_output_0_quantized : [4]usize = [_]usize{ 1, 512, 6, 6} ;
    var tensor__model_backbone_features_8_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_8_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_8_concat_output_0_quantized.deinit();

   // Step 41: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_8_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_8_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_8_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_8_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_8_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_8_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_8_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_8_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_8_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_8_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_8_conv_list_1_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 6, 6} ;
    var tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 42: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_8_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_8_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_593_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_593_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_594_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_8_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 6, 6} ;
    var tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 43: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_596_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_596_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_597_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 44: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_599_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_599_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_600_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 45: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_602_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_602_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_603_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_9_concat_output_0_quantized : [4]usize = [_]usize{ 1, 512, 6, 6} ;
    var tensor__model_backbone_features_9_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_9_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_9_concat_output_0_quantized.deinit();

   // Step 46: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_9_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_9_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_9_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_9_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_9_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_9_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_9_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_9_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_9_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_9_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_9_conv_list_2_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 6, 6} ;
    var tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 47: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_9_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_9_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_605_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_605_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_606_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_9_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 6, 6} ;
    var tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 48: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_608_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_608_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_609_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 49: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_611_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_611_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_612_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 64, 6, 6} ;
    var tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 50: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_614_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_614_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_615_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_10_concat_output_0_quantized : [4]usize = [_]usize{ 1, 512, 6, 6} ;
    var tensor__model_backbone_features_10_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_10_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_10_concat_output_0_quantized.deinit();

   // Step 51: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_10_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_10_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_10_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_10_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_10_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_10_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_10_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_10_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_10_conv_list_3_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_10_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_10_conv_list_0_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 512, 6, 6} ;
    var tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 52: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_10_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_10_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_617_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_617_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_618_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_10_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_11_skip_averagepool_output_0_quantized : [4]usize = [_]usize{ 1, 512, 3, 3} ;
    var tensor__model_backbone_features_11_skip_averagepool_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_11_skip_averagepool_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_11_skip_averagepool_output_0_quantized.deinit();

   // Step 53: qlinearaveragepool operation

    // QLinearAveragePool attributes
    var kernel_shape__model_backbone_features_11_skip_averagepool_output_0_quantized = [_]usize{3, 3};
    var strides__model_backbone_features_11_skip_averagepool_output_0_quantized = [_]usize{2, 2};
    var dilations__model_backbone_features_11_skip_averagepool_output_0_quantized = [_]usize{1, 1};
    var pads__model_backbone_features_11_skip_averagepool_output_0_quantized = [_]usize{1, 1, 1, 1};
    const auto_pad__model_backbone_features_11_skip_averagepool_output_0_quantized = tensMath.AutoPadType.NOTSET;

    // Perform QLinearAveragePool
    tensMath.lean_qlinearaveragepool(
        u8,
        f32,
        u8,
        @constCast(&tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized),
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_scale),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        @constCast(&param_lib.tensor__model_backbone_features_11_skip_averagepool_output_0_scale),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        &tensor__model_backbone_features_11_skip_averagepool_output_0_quantized,
        &kernel_shape__model_backbone_features_11_skip_averagepool_output_0_quantized,
        &strides__model_backbone_features_11_skip_averagepool_output_0_quantized,
        &dilations__model_backbone_features_11_skip_averagepool_output_0_quantized,
        &pads__model_backbone_features_11_skip_averagepool_output_0_quantized,
        auto_pad__model_backbone_features_11_skip_averagepool_output_0_quantized,
        true,
    ) catch return -1;


var shape_tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_quantized : [4]usize = [_]usize{ 1, 512, 3, 3} ;
    var tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_quantized.deinit();

   // Step 54: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_620_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_620_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_621_quantized), // bias
        &[_]usize{2,2}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        512, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_11_conv_list_0_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 3, 3} ;
    var tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 55: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_623_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_623_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_624_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_11_avd_layer_avd_layer_0_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 3, 3} ;
    var tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 56: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_626_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_626_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_627_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 3, 3} ;
    var tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 57: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_629_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_629_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_630_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_11_concat_output_0_quantized : [4]usize = [_]usize{ 1, 1024, 3, 3} ;
    var tensor__model_backbone_features_11_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_11_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_11_concat_output_0_quantized.deinit();

   // Step 58: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_11_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_11_skip_averagepool_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_11_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_11_skip_averagepool_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_11_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_11_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_11_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_11_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_11_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_11_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_11_skip_averagepool_output_0_quantized.deinit();
    tensor__model_backbone_features_11_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_11_conv_list_3_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 512, 3, 3} ;
    var tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 59: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_11_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_11_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_632_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_632_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_633_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_11_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 3, 3} ;
    var tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 60: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_635_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_635_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_636_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 3, 3} ;
    var tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 61: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_638_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_638_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_639_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 3, 3} ;
    var tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 62: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_641_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_641_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_642_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_12_concat_output_0_quantized : [4]usize = [_]usize{ 1, 1024, 3, 3} ;
    var tensor__model_backbone_features_12_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_12_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_12_concat_output_0_quantized.deinit();

   // Step 63: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_12_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_12_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_12_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_12_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_12_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_12_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_12_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_12_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_12_conv_list_0_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_12_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_12_conv_list_3_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 512, 3, 3} ;
    var tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized.deinit();

   // Step 64: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_12_concat_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_12_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_644_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_644_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_645_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_backbone_features_12_concat_output_0_quantized.deinit();


var shape_tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 256, 3, 3} ;
    var tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized.deinit();

   // Step 65: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_647_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_647_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_648_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 3, 3} ;
    var tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized.deinit();

   // Step 66: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_650_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_650_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_651_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_quantized : [4]usize = [_]usize{ 1, 128, 3, 3} ;
    var tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_quantized.deinit();

   // Step 67: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_onnx__conv_653_quantized), // w
        @constCast(&param_lib.tensor_onnx__conv_653_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_onnx__conv_654_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{1,1,1,1}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;

var shape_tensor__model_backbone_features_13_concat_output_0_quantized : [4]usize = [_]usize{ 1, 1024, 3, 3} ;
    var tensor__model_backbone_features_13_concat_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_backbone_features_13_concat_output_0_quantized) catch return -2;
    defer tensor__model_backbone_features_13_concat_output_0_quantized.deinit();

   // Step 68: qlinearconcat operation

    // Create arrays for QLinearConcat inputs
    var qlinearconcat_inputs__model_backbone_features_13_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized))), @as(*const Tensor(u8), @ptrCast(@constCast(&tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_quantized)))};

    var qlinearconcat_scales__model_backbone_features_13_concat_output_0_quantized = [_]*const Tensor(f32){@as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_scale))), @as(*const Tensor(f32), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_scale)))};

    var qlinearconcat_zero_points__model_backbone_features_13_concat_output_0_quantized = [_]*const Tensor(u8){@as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))), @as(*const Tensor(u8), @ptrCast(@constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point)))};
    // Perform QLinearConcat
    tensMath.lean_qlinearconcat(
        u8,
        f32,
        u8,
        &qlinearconcat_inputs__model_backbone_features_13_concat_output_0_quantized,
        &qlinearconcat_scales__model_backbone_features_13_concat_output_0_quantized,
        &qlinearconcat_zero_points__model_backbone_features_13_concat_output_0_quantized,
        @constCast(@as(*const Tensor(f32), @ptrCast(&param_lib.tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_scale))),
        @constCast(@as(*const Tensor(u8), @ptrCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point))),
        1,
        &tensor__model_backbone_features_13_concat_output_0_quantized,
    ) catch { tensor__model_backbone_features_13_concat_output_0_quantized.deinit(); return -1; };    tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_13_conv_list_1_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_13_conv_list_0_conv_conv_output_0_quantized.deinit();
    tensor__model_backbone_features_13_conv_list_3_conv_conv_output_0_quantized.deinit();


var shape_tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_quantized : [4]usize = [_]usize{ 1, 1024, 1, 1} ;
    var tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_quantized) catch return -2;
    defer tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_quantized.deinit();

   // Step 69: qlinearglobalaveragepool operation

    tensMath.qlinearglobalaveragepool_lean(
        @constCast(&tensor__model_backbone_features_13_concat_output_0_quantized),
        @constCast(&param_lib.tensor__model_backbone_features_13_conv_list_2_conv_conv_output_0_scale),
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point),
        &tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_quantized,
        @constCast(&param_lib.tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_scale),
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point),
    ) catch return -1;
    tensor__model_backbone_features_13_concat_output_0_quantized.deinit();


var shape_tensor__model_cls_head_classifier_classifier_2_conv_output_0_quantized : [4]usize = [_]usize{ 1, 80, 1, 1} ;
    var tensor__model_cls_head_classifier_classifier_2_conv_output_0_quantized = Tensor(u8).fromShape(&allocator, &shape_tensor__model_cls_head_classifier_classifier_2_conv_output_0_quantized) catch return -2;
    defer tensor__model_cls_head_classifier_classifier_2_conv_output_0_quantized.deinit();

   // Step 70: qlinearconv operation
    tensMath.qlinearconv_dispatch(
        u8, // InputType
        i8, // WeightType
        f32, // ScaleType
        u8, // OutputType
        i32, // BiasType
        @constCast(&tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_quantized), // input x
        @constCast(&param_lib.tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_scale), // x_scale
        @constCast(&param_lib.tensor__model_backbone_features_0_conv_conv_output_0_zero_point), // x_zero_point
        @constCast(&param_lib.tensor_model_cls_head_classifier_2_weight_quantized), // w
        @constCast(&param_lib.tensor_model_cls_head_classifier_2_weight_scale), // w_scale
        @constCast(&param_lib.tensor_onnx__conv_497_zero_point), // w_zero_point
        &tensor__model_cls_head_classifier_classifier_2_conv_output_0_quantized, // output
        @constCast(&param_lib.tensor__model_cls_head_classifier_classifier_2_conv_output_0_scale), // y_scale
        @constCast(&param_lib.tensor__model_cls_head_classifier_classifier_2_conv_output_0_zero_point), // y_zero_point
        @constCast(&param_lib.tensor_model_cls_head_classifier_2_bias_quantized), // bias
        &[_]usize{1,1}, // stride
        &[_]usize{0,0,0,0}, // pads
        &[_]usize{1,1}, // dilations
        1, // group
        "NOTSET", // auto_pad
    ) catch return -1;    tensor__model_cls_head_classifier_classifier_0_globalaveragepool_output_0_quantized.deinit();


var shape_tensor__model_cls_head_classifier_classifier_2_conv_output_0 : [4]usize = [_]usize{ 1, 80, 1, 1} ;
    var tensor__model_cls_head_classifier_classifier_2_conv_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor__model_cls_head_classifier_classifier_2_conv_output_0) catch return -2;
    defer tensor__model_cls_head_classifier_classifier_2_conv_output_0.deinit();

   // Step 71: dequantizelinear operation


    tensMath.dequantizeLinear_lean(u8, // InputType
                                 f32, // OutputType
                                 u8, // ZeroPointType
                                 &tensor__model_cls_head_classifier_classifier_2_conv_output_0_quantized, // x: input tensor
                                 @constCast(&param_lib.tensor__model_cls_head_classifier_classifier_2_conv_output_0_scale), // x_scale
                                 @constCast(&param_lib.tensor__model_cls_head_classifier_classifier_2_conv_output_0_zero_point), // x_zero_point
                                 1,  // axis
                                 0,  // block_size
                                 &tensor__model_cls_head_classifier_classifier_2_conv_output_0, // y: output tensor
    ) catch return -1;    tensor__model_cls_head_classifier_classifier_2_conv_output_0_quantized.deinit();


var shape_tensor_logits : [2]usize = [_]usize{ 1, 80} ;
    var tensor_logits = Tensor(f32).fromShape(&allocator, &shape_tensor_logits) catch return -2;

   // Step 72: flatten operation


    tensMath.flatten_lean(f32, &tensor__model_cls_head_classifier_classifier_2_conv_output_0, &tensor_logits) catch return -1;    tensor__model_cls_head_classifier_classifier_2_conv_output_0.deinit();
     
     const output_zant_slice = allocator.alloc(T_out, tensor_logits.size) catch return -3;
     @memcpy(output_zant_slice, tensor_logits.data[0..tensor_logits.size]);
     
     // Deallocate the output tensor after copying its data
     tensor_logits.deinit();
      
     //The Caller must handle the memory of output_zant_slice
     result.* = output_zant_slice.ptr;

    return 0;

}