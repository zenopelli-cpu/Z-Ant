
 const Tensor = @import("tensor").Tensor;
 const tensMath = @import("tensor_math");
 const pkgAllocator = @import("pkgAllocator");
 const allocator = pkgAllocator.allocator;
 const utils = @import("codeGen_utils.zig");

var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;

pub export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
}


 var buf: [4096 * 10]u8 = undefined;
 var fba_state = @import("std").heap.FixedBufferAllocator.init(&buf);
 const fba = fba_state.allocator();
 const T = f32;

 // ---------------------------------------------------
 // +         Initializing Weights and Biases         +
 // ---------------------------------------------------

 // ----------- Initializing tensor_layer1_weight;

const shape_tensor_layer1_weight : [2]usize = [_]usize{  5,  5} ;
const array_layer1_weight : [25]f32 = [_]f32{ -7.995443e-2, -3.7083545e-1, -2.2724475e-1, -2.4103428e-1, -1.6814743e-1, 2.641146e-1, -1.7869543e-1, 3.719517e-1, 1.5944983e-1, -4.3663558e-1, -6.7542054e-2, 3.215769e-1, -2.5744948e-1, 2.9138434e-1, 3.0216023e-1, 6.7657e-2, 4.3623355e-1, 3.681088e-1, 1.1790166e-2, -2.933837e-1, -1.0971639e-1, -4.526883e-2, 4.082875e-1, 9.778068e-2, -3.2061434e-1} ;
const tensor_layer1_weight = Tensor(f32).fromConstBuffer( &array_layer1_weight, &shape_tensor_layer1_weight);

 // ----------- Initializing tensor_layer1_bias;

const shape_tensor_layer1_bias : [1]usize = [_]usize{  5} ;
const array_layer1_bias : [5]f32 = [_]f32{ 1.190428e-1, 2.4293907e-1, -3.8535364e-2, 1.3539758e-1, -1.00039355e-1} ;
const tensor_layer1_bias = Tensor(f32).fromConstBuffer( &array_layer1_bias, &shape_tensor_layer1_bias);

 // ----------- Initializing tensor_layer2_weight;

const shape_tensor_layer2_weight : [2]usize = [_]usize{  5,  5} ;
const array_layer2_weight : [25]f32 = [_]f32{ 6.443466e-2, 3.8758186e-1, -1.0957069e-1, 3.2179978e-1, 5.5207577e-2, 1.1797485e-1, 2.8706074e-1, -2.810581e-1, 4.0451333e-1, 3.9878613e-1, -2.561895e-1, -4.1911042e-1, 4.5665205e-2, 1.2990884e-1, -5.8087703e-2, -4.1650724e-1, -2.770127e-1, 3.9719265e-2, -4.4663292e-1, 2.4146792e-1, -2.64401e-1, -2.1230811e-1, -7.49117e-2, -2.404422e-1, -1.2177584e-2} ;
const tensor_layer2_weight = Tensor(f32).fromConstBuffer( &array_layer2_weight, &shape_tensor_layer2_weight);

 // ----------- Initializing tensor_layer2_bias;

const shape_tensor_layer2_bias : [1]usize = [_]usize{  5} ;
const array_layer2_bias : [5]f32 = [_]f32{ 3.573904e-1, 3.3577403e-1, 1.3116845e-1, -3.886374e-1, -2.9121396e-1} ;
const tensor_layer2_bias = Tensor(f32).fromConstBuffer( &array_layer2_bias, &shape_tensor_layer2_bias);

 // ---------------------------------------------------
 // +         Initializing output Tensors             +
 // ---------------------------------------------------

var shape_tensor__layer1_gemm_output_0 : [1]usize = [_]usize{ 5} ;
var array__layer1_gemm_output_0 = undefined;
var tensor__layer1_gemm_output_0 = Tensor(T).fromConstBuffer( &array__layer1_gemm_output_0, &shape_tensor__layer1_gemm_output_0);

var shape_tensor__relu_relu_output_0 : [1]usize = [_]usize{ 5} ;
var array__relu_relu_output_0 = undefined;
var tensor__relu_relu_output_0 = Tensor(T).fromConstBuffer( &array__relu_relu_output_0, &shape_tensor__relu_relu_output_0);

var shape_tensor__layer2_gemm_output_0 : [1]usize = [_]usize{ 5} ;
var array__layer2_gemm_output_0 = undefined;
var tensor__layer2_gemm_output_0 = Tensor(T).fromConstBuffer( &array__layer2_gemm_output_0, &shape_tensor__layer2_gemm_output_0);

var shape_tensor__relu_1_relu_output_0 : [1]usize = [_]usize{ 5} ;
var array__relu_1_relu_output_0 = undefined;
var tensor__relu_1_relu_output_0 = Tensor(T).fromConstBuffer( &array__relu_1_relu_output_0, &shape_tensor__relu_1_relu_output_0);

var shape_tensor_output : [1]usize = [_]usize{ 5} ;
var array_output = undefined;
var tensor_output = Tensor(T).fromConstBuffer( &array_output, &shape_tensor_output);


export fn predict( 
    input: [*]T,
    input_shape: [*]u32,
    shape_len: u32,
    result: *[*]T,
) void {

    if (log_function) |log| {
        log(@constCast(@ptrCast("Starting prediction...\n")));
    }
    //checks on the input parameters
    if (shape_len == 0) return ;
    if(shape_len != 4) return ;
    if( input_shape[0] != 1) return ;
    if( input_shape[1] != 2) return ;
    if( input_shape[2] != 3) return ;
    if( input_shape[3] != 4) return ;  
    //computing the size of the input tensor
    var size: u32 = 1;
    for(0..shape_len) |dim_i| {
        size *= input_shape[dim_i];
    }
     
    //allocating space in memory for the data
    const data = allocator.alloc(T, size) catch return;
    for (0..size) |i| {
        data[i] = input[i]; // Copying input elements 
    }
    
    //converting the shape from [*]u32 to []usize
    const usized_shape: []usize = utils.u32ToUsize(input_shape, shape_len) catch return;
    var tensor_input = Tensor(T).fromShape(&allocator, @constCast(usized_shape)) catch return;
    @memcpy(tensor_input.data, data);
    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Gemm operation...\n")));
    }
    tensMath.gemm_lean(T, &tensor_input, @constCast(&tensor_layer1_weight) , @constCast(&tensor_layer1_bias), 1e0, 1e0, false, false, &tensor__layer1_gemm_output_0 ) catch return;
    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Relu operation...\n")));
    }
    tensMath.ReLU_lean(T, &tensor__layer1_gemm_output_0, &tensor__relu_relu_output_0) catch return;
    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Gemm operation...\n")));
    }
    tensMath.gemm_lean(T, &tensor__relu_relu_output_0, @constCast(&tensor_layer2_weight) , @constCast(&tensor_layer2_bias), 1e0, 1e0, false, false, &tensor__layer2_gemm_output_0 ) catch return;
    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Relu operation...\n")));
    }
    tensMath.ReLU_lean(T, &tensor__layer2_gemm_output_0, &tensor__relu_1_relu_output_0) catch return;
    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Softmax operation...\n")));
    }
    tensMath.softmax_lean(T, &tensor__relu_1_relu_output_0, &tensor_output) catch return;
    result.* = tensor_output.data.ptr;

    if (log_function) |log| {
        log(@constCast(@ptrCast("Prediction completed.\n")));
    }
} 