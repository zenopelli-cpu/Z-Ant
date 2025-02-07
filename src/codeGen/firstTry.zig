
 const std = @import("std");
 const Tensor = @import("tensor").Tensor;
 const TensMath = @import("tensor_m");
 const pkgAllocator = @import("pkgAllocator");

 const allocator = pkgAllocator.allocator;

 // ----------- initializing tensor_layer1_weight;

const shape_tensor_layer1_weight : [2]usize = [_]usize{ 5, 5} ;
const array_layer1_weight : [25]f32 = [_]f32{ -7.995443e-2, -3.7083545e-1, -2.2724475e-1, -2.4103428e-1, -1.6814743e-1, 2.641146e-1, -1.7869543e-1, 3.719517e-1, 1.5944983e-1, -4.3663558e-1, -6.7542054e-2, 3.215769e-1, -2.5744948e-1, 2.9138434e-1, 3.0216023e-1, 6.7657e-2, 4.3623355e-1, 3.681088e-1, 1.1790166e-2, -2.933837e-1, -1.0971639e-1, -4.526883e-2, 4.082875e-1, 9.778068e-2, -3.2061434e-1} ;
const tensor_layer1_weight = Tensor(f32).fromArray(&allocator, & array_layer1_weight, &shape_tensor_layer1_weight);

 // ----------- initializing tensor_layer1_bias;

const shape_tensor_layer1_bias : [1]usize = [_]usize{ 5} ;
const array_layer1_bias : [5]f32 = [_]f32{ 1.190428e-1, 2.4293907e-1, -3.8535364e-2, 1.3539758e-1, -1.00039355e-1} ;
const tensor_layer1_bias = Tensor(f32).fromArray(&allocator, & array_layer1_bias, &shape_tensor_layer1_bias);

 // ----------- initializing tensor_layer2_weight;

const shape_tensor_layer2_weight : [2]usize = [_]usize{ 5, 5} ;
const array_layer2_weight : [25]f32 = [_]f32{ 6.443466e-2, 3.8758186e-1, -1.0957069e-1, 3.2179978e-1, 5.5207577e-2, 1.1797485e-1, 2.8706074e-1, -2.810581e-1, 4.0451333e-1, 3.9878613e-1, -2.561895e-1, -4.1911042e-1, 4.5665205e-2, 1.2990884e-1, -5.8087703e-2, -4.1650724e-1, -2.770127e-1, 3.9719265e-2, -4.4663292e-1, 2.4146792e-1, -2.64401e-1, -2.1230811e-1, -7.49117e-2, -2.404422e-1, -1.2177584e-2} ;
const tensor_layer2_weight = Tensor(f32).fromArray(&allocator, & array_layer2_weight, &shape_tensor_layer2_weight);

 // ----------- initializing tensor_layer2_bias;

const shape_tensor_layer2_bias : [1]usize = [_]usize{ 5} ;
const array_layer2_bias : [5]f32 = [_]f32{ 3.573904e-1, 3.3577403e-1, 1.3116845e-1, -3.886374e-1, -2.9121396e-1} ;
const tensor_layer2_bias = Tensor(f32).fromArray(&allocator, & array_layer2_bias, &shape_tensor_layer2_bias);


pub fn predict(comptime T: anytype, tensor_input: Tensor(T)) {

const shape_tensor__layer1_Gemm_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor__layer1_Gemm_output_0 = Tensor(f32).fromArray(&allocator, & array__layer1_Gemm_output_0, &shape_tensor__layer1_Gemm_output_0);
 //forwarding operation : Gemm
 //parameters:
 //   inputs: 
//      -> input 
//      -> layer1.weight 
//      -> layer1.bias 
//    outputs: 
//      <- /layer1/Gemm_output_0 

const shape_tensor__relu_Relu_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor__relu_Relu_output_0 = Tensor(f32).fromArray(&allocator, & array__relu_Relu_output_0, &shape_tensor__relu_Relu_output_0);
 //forwarding operation : Relu
 //parameters:
 //   inputs: 
//      -> /layer1/Gemm_output_0 
//    outputs: 
//      <- /relu/Relu_output_0 

const shape_tensor__layer2_Gemm_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor__layer2_Gemm_output_0 = Tensor(f32).fromArray(&allocator, & array__layer2_Gemm_output_0, &shape_tensor__layer2_Gemm_output_0);
 //forwarding operation : Gemm
 //parameters:
 //   inputs: 
//      -> /relu/Relu_output_0 
//      -> layer2.weight 
//      -> layer2.bias 
//    outputs: 
//      <- /layer2/Gemm_output_0 

const shape_tensor__relu_1_Relu_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor__relu_1_Relu_output_0 = Tensor(f32).fromArray(&allocator, & array__relu_1_Relu_output_0, &shape_tensor__relu_1_Relu_output_0);
 //forwarding operation : Relu
 //parameters:
 //   inputs: 
//      -> /layer2/Gemm_output_0 
//    outputs: 
//      <- /relu_1/Relu_output_0 

const shape_tensor_output : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_output = Tensor(f32).fromArray(&allocator, & array_output, &shape_tensor_output);
 //forwarding operation : Softmax
 //parameters:
 //   inputs: 
//      -> /relu_1/Relu_output_0 
//    outputs: 
//      <- output 
}