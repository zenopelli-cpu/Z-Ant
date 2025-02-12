
 const std = @import("std");
 const Tensor = @import("tensor").Tensor;
 const TensMath = @import("lean_tensor_math");
 const pkgAllocator = @import("pkgAllocator");
 const allocator = pkgAllocator.allocator;

 // ---------------------------------------------------
 // +         Initializing Weights and Biases         +
 // ---------------------------------------------------

 // ---------------------------------------------------
 // +         initializing output Tensors             +
 // ---------------------------------------------------

const shape_tensor_constant377 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_constant377 = Tensor(f32).fromShape(&allocator, &shape_tensor_constant377);

const shape_tensor_block386_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_block386_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_block386_output_0);

const shape_tensor_constant321 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_constant321 = Tensor(f32).fromShape(&allocator, &shape_tensor_constant321);

const shape_tensor_convolution396_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_convolution396_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_convolution396_output_0);

const shape_tensor_constant318 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_constant318 = Tensor(f32).fromShape(&allocator, &shape_tensor_constant318);

const shape_tensor_reshape398_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_reshape398_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_reshape398_output_0);

const shape_tensor_reshape398_output_0_reshape1 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_reshape398_output_0_reshape1 = Tensor(f32).fromShape(&allocator, &shape_tensor_reshape398_output_0_reshape1);

const shape_tensor_plus400_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_plus400_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_plus400_output_0);

const shape_tensor_relu402_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_relu402_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_relu402_output_0);

const shape_tensor_pooling404_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_pooling404_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_pooling404_output_0);

const shape_tensor_constant340 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_constant340 = Tensor(f32).fromShape(&allocator, &shape_tensor_constant340);

const shape_tensor_convolution406_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_convolution406_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_convolution406_output_0);

const shape_tensor_constant346 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_constant346 = Tensor(f32).fromShape(&allocator, &shape_tensor_constant346);

const shape_tensor_reshape408_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_reshape408_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_reshape408_output_0);

const shape_tensor_reshape408_output_0_reshape1 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_reshape408_output_0_reshape1 = Tensor(f32).fromShape(&allocator, &shape_tensor_reshape408_output_0_reshape1);

const shape_tensor_plus410_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_plus410_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_plus410_output_0);

const shape_tensor_relu412_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_relu412_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_relu412_output_0);

const shape_tensor_pooling414_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_pooling414_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_pooling414_output_0);

const shape_tensor_reshape416_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_reshape416_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_reshape416_output_0);

const shape_tensor_constant312 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_constant312 = Tensor(f32).fromShape(&allocator, &shape_tensor_constant312);

const shape_tensor_reshape393_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_reshape393_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_reshape393_output_0);

const shape_tensor_times418_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_times418_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_times418_output_0);

const shape_tensor_constant367 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_constant367 = Tensor(f32).fromShape(&allocator, &shape_tensor_constant367);

const shape_tensor_plus422_output_0 : [4]usize = [_]usize{1, 1, 1, 1} ;
const tensor_plus422_output_0 = Tensor(f32).fromShape(&allocator, &shape_tensor_plus422_output_0);


pub fn predict(comptime T: anytype, tensor_input: Tensor(T)) !void {

    //forwarding operation : Constant
    //parameters:
    //   inputs: 
    //    outputs: 
    //      <- Constant377 // Handle Constant


    //forwarding operation : Constant
    //parameters:
    //   inputs: 
    //    outputs: 
    //      <- Constant321 // Handle Constant


    //forwarding operation : Constant
    //parameters:
    //   inputs: 
    //    outputs: 
    //      <- Constant318 // Handle Constant


    //forwarding operation : Constant
    //parameters:
    //   inputs: 
    //    outputs: 
    //      <- Constant340 // Handle Constant


    //forwarding operation : Constant
    //parameters:
    //   inputs: 
    //    outputs: 
    //      <- Constant346 // Handle Constant


    //forwarding operation : Constant
    //parameters:
    //   inputs: 
    //    outputs: 
    //      <- Constant312 // Handle Constant


    //forwarding operation : Constant
    //parameters:
    //   inputs: 
    //    outputs: 
    //      <- Constant367 // Handle Constant


    //forwarding operation : Div
    //parameters:
    //   inputs: 
    //      -> input 
    //      -> Constant377 
    //    outputs: 
    //      <- Block386_Output_0 // Handle Div


    //forwarding operation : Reshape
    //parameters:
    //   inputs: 
    //      -> Constant318 
    //    outputs: 
    //      <- Reshape398_Output_0 

    //forwarding operation : Reshape
    //parameters:
    //   inputs: 
    //      -> Constant346 
    //    outputs: 
    //      <- Reshape408_Output_0 

    //forwarding operation : Reshape
    //parameters:
    //   inputs: 
    //      -> Constant312 
    //    outputs: 
    //      <- Reshape393_Output_0 

    //forwarding operation : Conv
    //parameters:
    //   inputs: 
    //      -> Block386_Output_0 
    //      -> Constant321 
    //    outputs: 
    //      <- Convolution396_Output_0 // Handle Conv


    //forwarding operation : Reshape
    //parameters:
    //   inputs: 
    //      -> Reshape398_Output_0 
    //    outputs: 
    //      <- Reshape398_Output_0_reshape1 

    //forwarding operation : Reshape
    //parameters:
    //   inputs: 
    //      -> Reshape408_Output_0 
    //    outputs: 
    //      <- Reshape408_Output_0_reshape1 

    //forwarding operation : Add
    //parameters:
    //   inputs: 
    //      -> Convolution396_Output_0 
    //      -> Reshape398_Output_0_reshape1 
    //    outputs: 
    //      <- Plus400_Output_0 // Handle Add


    //forwarding operation : Relu
    //parameters:
    //   inputs: 
    //      -> Plus400_Output_0 
    //    outputs: 
    //      <- ReLU402_Output_0 // Handle Relu


    //forwarding operation : MaxPool
    //parameters:
    //   inputs: 
    //      -> ReLU402_Output_0 
    //    outputs: 
    //      <- Pooling404_Output_0 // Handle MaxPool


    //forwarding operation : Conv
    //parameters:
    //   inputs: 
    //      -> Pooling404_Output_0 
    //      -> Constant340 
    //    outputs: 
    //      <- Convolution406_Output_0 // Handle Conv


    //forwarding operation : Add
    //parameters:
    //   inputs: 
    //      -> Convolution406_Output_0 
    //      -> Reshape408_Output_0_reshape1 
    //    outputs: 
    //      <- Plus410_Output_0 // Handle Add


    //forwarding operation : Relu
    //parameters:
    //   inputs: 
    //      -> Plus410_Output_0 
    //    outputs: 
    //      <- ReLU412_Output_0 // Handle Relu


    //forwarding operation : MaxPool
    //parameters:
    //   inputs: 
    //      -> ReLU412_Output_0 
    //    outputs: 
    //      <- Pooling414_Output_0 // Handle MaxPool


    //forwarding operation : Reshape
    //parameters:
    //   inputs: 
    //      -> Pooling414_Output_0 
    //    outputs: 
    //      <- Reshape416_Output_0 

    //forwarding operation : MatMul
    //parameters:
    //   inputs: 
    //      -> Reshape416_Output_0 
    //      -> Reshape393_Output_0 
    //    outputs: 
    //      <- Times418_Output_0 // Handle MatMul


    //forwarding operation : Add
    //parameters:
    //   inputs: 
    //      -> Times418_Output_0 
    //      -> Constant367 
    //    outputs: 
    //      <- Plus422_Output_0 // Handle Add

 }