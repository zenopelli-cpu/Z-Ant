//! Tensor math contains all the functions to perform operations on tensors
const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const Architectures = @import("architectures").Architectures; //Import Architectures type
const Converter = @import("typeC");
const Layer = @import("Layer");

const PoolingType = @import("poolingLayer").PoolingType;

//import error libraries
const TensorMathError = @import("errorHandler").TensorMathError;
const ArchitectureError = @import("errorHandler").ArchitectureError;
const TensorError = @import("errorHandler").TensorError;

const pkg_allocator = @import("pkgAllocator").allocator;

/// Function that add the bias for all the features in the tensor
pub fn add_bias(comptime T: anytype, tensor: *Tensor(T), bias: *Tensor(T)) !void {
    // Checks:
    if (tensor.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.shape.len != 1) {
        return TensorMathError.InputTensorsWrongShape;
    }
    const len = bias.shape[0];
    if (len != tensor.shape[tensor.shape.len - 1]) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // Allocate an array for threads, one for each row of the tensor
    const num_threads = tensor.size / bias.size;

    var threads = try pkg_allocator.alloc(std.Thread, num_threads); //Array to save thread handles

    var index: usize = 0;
    var i: usize = 0;

    // Start a thread for each row of the tensor
    while (index < tensor.size) : (i += 1) {
        threads[i] = try std.Thread.spawn(.{}, add_bias_thread, .{ T, tensor.data, index, len, bias });
        index += len;
    }

    // Merges all threads
    for (threads) |*thread| {
        thread.join(); // Use try to catch any errors
    }

    // Free the thread array
    pkg_allocator.free(threads);
}

fn add_bias_thread(comptime T: anytype, array: []T, start: usize, len: usize, bias: *Tensor(T)) void {
    for (0..len) |i| {
        array[start + i] += bias.data[i];
    }
}
/// Performs the mean of a given tensor. It is a reduction operation, collapsing the whole tenosr into a single value.
pub fn mean(comptime T: anytype, tensor: *Tensor(T)) f32 {
    var res: f32 = 0;

    for (tensor.data) |*d| {
        res += Converter.convert(T, f32, d.*);
    }
    res = res / Converter.convert(usize, f32, tensor.size);
    return res;
}

///Returns a Tensor with the same shape pf t1 and t2, where each element --> out[location] = t1[location] + t2[location]
pub fn sum_tensors(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !Tensor(Tout) {

    //selecting between all possible architectures
    return switch (arch) {
        Architectures.CPU => return CPU_sum_tensors(Tin, Tout, t1, t2),

        Architectures.GPU => {
            std.debug.print("{} is under developement \n", .{arch});
            return ArchitectureError.UnderDevelopementArchitecture;
        },
        Architectures.SP32 => {
            std.debug.print("{} is under developement \n", .{arch});
            return ArchitectureError.UnderDevelopementArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}

//Return the sum of the tensors inside another Tensor (t3)
fn CPU_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    if (t1.size != t2.size) return TensorMathError.InputTensorDifferentSize;

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    }

    // Allocating the array for the sum
    var out_sum = try t1.allocator.alloc(outputType, t1.size);
    defer t1.allocator.free(out_sum); // Ensure out_sum gets freed in case of error

    var i: usize = 0;
    const unroll_factor: usize = 4;

    // Loop unrolling
    while (i + unroll_factor <= t1.size) : (i += 4) {
        out_sum[i] = t1.data[i] + t2.data[i];
        out_sum[i + 1] = t1.data[i + 1] + t2.data[i + 1];
        out_sum[i + 2] = t1.data[i + 2] + t2.data[i + 2];
        out_sum[i + 3] = t1.data[i + 3] + t2.data[i + 3];
    }

    // Handle any remaining elements
    while (i < t1.size) : (i += 1) {
        out_sum[i] = t1.data[i] + t2.data[i];
    }

    // Create output tensor
    const out_tensor = try Tensor(outputType).fromArray(t1.allocator, out_sum, t1.shape);

    // Remove the defer since the tensor will manage its own memory after creation
    return out_tensor;
}

// DOT PRODUCT -----------------------------------------------------------------------------------------------------------------------

/// Returns the dot product of two tensors. The dot product is the sum of the products of the corresponding entries of the two sequences of numbers.
/// Deprecated: use dot_product_tensor instead
pub fn compute_dot_product(comptime T: type, input: *Tensor(T), weights: *Tensor(T)) !Tensor(T) {
    return try CPU_dot_product_tensors(T, T, input, weights);
}

/// Returns the dot product of two tensors. The dot product is the sum of the products of the corresponding entries of the two sequences of numbers.
pub fn dot_product_tensor(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !Tensor(Tout) {
    return switch (arch) {
        Architectures.CPU => return CPU_dot_product_tensors(Tin, Tout, t1, t2),
        Architectures.GPU => {
            std.debug.print("{} is under development\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        Architectures.SP32 => {
            std.debug.print("{} is under development\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}
/// Implementation of dot product for CPU architecture still not parallelized
pub fn CPU_dot_product_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {

    //CHECKS :
    const nDimT1 = t1.shape.len; //number of dimesion of tensor 1
    const nDimT2 = t2.shape.len; //number of dimesion of tensor 2
    // -imput shape:
    if (nDimT1 != nDimT2) return TensorMathError.InputTensorDifferentShape;

    //-dimensional compatibility:
    // If you have two matrices A and B, to compute the product A×B, the number of columns in A must be equal to the number of rows in B.
    // If A is a matrix of dimensions m×n and B is a matrix of dimensions n×p, then the product A×B is defined, and it results in a matrix of dimensions m×p.
    if (t1.shape[nDimT1 - 1] != t2.shape[nDimT1 - 2]) return TensorMathError.InputTensorsWrongShape;

    // -this check is necassary to avoid loss of information/ overflow when working with quantized tensors
    // usually quantization reduce to a maximum of 16bit, to the next check is divided between quant and non-quant data
    //bool (1 bit)
    // u1 (1 bit)
    // i8 (8 bits)
    // u8 (8 bits)
    // i16 (16 bits)
    // u16 (16 bits)
    // f16 (16 bits)
    // i32 (32 bits)
    // u32 (32 bits)
    // f32 (32 bits)
    // i64 (64 bits)
    // u64 (64 bits)
    // f64 (64 bits)
    // i128 (128 bits)
    // u128 (128 bits)
    // f128 (128 bits)
    if (@TypeOf(outputType) == @TypeOf(inputType)) {
        // Se input e output sono dello stesso tipo, non eseguire il controllo
        // Evitiamo l'errore in questo caso
    } else {
        if (@bitSizeOf(outputType) <= 16) { //quantized
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { //non-quant
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    //CREATING output_tensor :
    const allocator = pkg_allocator;
    var out_shape = try allocator.alloc(usize, nDimT1); //I had to use alloc() bacause nDimT1 is not known at comptime
    defer pkg_allocator.free(out_shape);
    //defining the resulting shape
    for (0..(nDimT1 - 2)) |i| {
        out_shape[i] = t1.shape[i];
    }
    out_shape[nDimT1 - 2] = t1.shape[nDimT1 - 2];
    out_shape[nDimT1 - 1] = t2.shape[nDimT1 - 1];

    var out_tensor = try Tensor(outputType).fromShape(&pkg_allocator, out_shape);
    try out_tensor.set(0, 0);
    //initialize the current location to all 0
    const location = try pkg_allocator.alloc(usize, nDimT1);
    defer pkg_allocator.free(location);
    for (location) |*loc| {
        loc.* = 0;
    }

    //call mutidim_mat_mul to handle multidimensionality
    try multidim_multiplication(
        inputType,
        outputType,
        t1,
        t2,
        &out_tensor,
        0,
        location,
    );
    //print output tensor shape

    return out_tensor;
}
/// Function that performs the multiplication of two tensors used in a recursive way to handle multidimensional tensors
fn multidim_multiplication(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), t3: *Tensor(outputType), current_depth: usize, location: []usize) !void {
    if (current_depth == (t1.shape.len - 2)) {

        //declaring sum
        var sum: outputType = 0;

        //with the first two for loop I iterate over t3
        for (0..t1.shape[current_depth]) |row| { //for each row of t1

            for (0..t2.shape[current_depth + 1]) |col| { //for each col of t2

                sum = 0;

                for (0..t1.shape[current_depth + 1]) |i| {

                    //compose the location on t1
                    location[t1.shape.len - 1] = i; //location
                    location[t1.shape.len - 2] = row; //location

                    //getting the correct numbers in t1
                    const a = try t1.get_at(location);

                    //compose the location on t2
                    location[t1.shape.len - 1] = col; //location
                    location[t1.shape.len - 2] = i; //location

                    //getting the correct numbers in t2
                    const b = try t2.get_at(location);

                    sum += a * b;
                }

                //compose the location on t3
                location[t1.shape.len - 1] = col; //col on the out tensor matrix
                location[t1.shape.len - 2] = row; //row on the out tensor matrix

                try t3.set_at(location, sum);
            }
        }
    } else {
        for (0..t1.shape[current_depth]) |element_at_current_depth| {
            //print location:
            //std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
            location[current_depth] = element_at_current_depth;
            //otherwise I have to go deeper
            try multidim_multiplication(
                inputType,
                outputType,
                t1,
                t2,
                t3,
                current_depth + 1,
                location,
            );
        }
    }
}

// CONVOLVE -----------------------------------------------------------------------------------------------------------------------

/// Multidim Conv
/// INPUT:
///     INPUT[input.shape.len - 4] -> batches
///     INPUT[input.shape.len - 3] -> input channels
///     INPUT[input.shape.len - 2] -> rows
///     INPUT[input.shape.len - 1] -> cols
/// KERNEL:
///     KERNEL[kernel.shape.len - 4] -> filters
///     KERNEL[kernel.shape.len - 3] -> channels
///     KERNEL[kernel.shape.len - 2] -> rows
///     KERNEL[kernel.shape.len - 1] -> cols
/// OUTPUT:
///     OUTPUT[output.shape.len - 4] -> input_batch
///     OUTPUT[output.shape.len - 3] -> output channels (number_of_kernel_filters)
///     OUTPUT[output.shape.len - 2] -> rows
///     OUTPUT[output.shape.len - 1] -> cols
fn multidim_convolution_with_bias(
    comptime inputType: anytype,
    comptime outputType: anytype,
    input: *Tensor(inputType),
    kernel: *Tensor(inputType),
    output: *Tensor(outputType),
    bias: *Tensor(outputType),
    stride: []usize,
    current_dim: usize, //represent the dimension we are currently working in
    location: []usize,
) !void {
    if (current_dim == input.shape.len - 3) { //
        //std.debug.print("\n\n KERNEL:{any} \n stride:{any} \n ", .{ kernel.data, stride });
        // std.debug.print("\n         input shape:{any} ", .{input.shape});
        // std.debug.print("\n         kernel shape:{any} ", .{kernel.shape});
        // std.debug.print("\n         output shape:{any} ", .{output.shape});
        // std.debug.print("\n         stride:{any} ", .{stride});

        const outDim = output.shape.len;
        const inDim = input.shape.len;
        const kernelDim = kernel.shape.len;
        const biasDim = bias.shape.len;

        const kernel_location = try pkg_allocator.alloc(usize, kernelDim); //coordinates in the kernel space
        defer pkg_allocator.free(kernel_location);
        const input_location = try pkg_allocator.alloc(usize, inDim); //coordinates in the input space
        defer pkg_allocator.free(input_location);
        const output_location = try pkg_allocator.alloc(usize, outDim); //coordinates in the output space
        defer pkg_allocator.free(output_location);
        const bias_location = try pkg_allocator.alloc(usize, biasDim); //coordinates in the bias space
        defer pkg_allocator.free(bias_location);

        //init kernel coordinates
        @memset(kernel_location, 0);
        for (0..kernelDim - 4) |i| { //copying the batches
            kernel_location[i] = input_location[i];
        }

        //init input coordinates
        @memcpy(input_location, location);

        //init output coordinates
        @memset(output_location, 0);
        for (0..outDim - 3) |i| { //copying the batches
            output_location[i] = input_location[i];
        }

        //init bias coordinates
        @memset(bias_location, 0);
        for (0..biasDim - 1) |i| { //copying the batches
            bias_location[i] = input_location[i];
        }

        // std.debug.print("multidim_convolution_with_bias: location: {d}, sum: {}\n", .{ location, sum });

        // for each filter in the kernel (remember, every filter must by applied to every input batch)
        for (0..kernel.shape[kernelDim - 4]) |filter_number| {
            output_location[outDim - 3] = filter_number; // set output channel
            kernel_location[kernelDim - 4] = filter_number; //set kernel filter
            bias_location[biasDim - 1] = filter_number; //set bias location, one bias for each kernel filter

            // for each row of the output
            for (0..output.shape[outDim - 2]) |out_row| {
                output_location[outDim - 2] = out_row;

                const startInputRow = out_row * stride[0]; //the corresponding starting imput row coordinates
                input_location[inDim - 2] = startInputRow; //set the starting input location

                // for each col of the output
                for (0..output.shape[outDim - 1]) |out_col| {
                    output_location[outDim - 1] = out_col;

                    const startInputCol = out_col * stride[1]; //the corresponding starting imput cols coordinates
                    input_location[inDim - 1] = startInputCol; //set the starting input location

                    var sum: inputType = 0;
                    // now iterate the same input submatrix in all the channels
                    for (0..kernel.shape[kernelDim - 3]) |channel| { // kernel channels
                        kernel_location[kernelDim - 3] = channel;
                        input_location[inDim - 3] = channel;

                        for (0..kernel.shape[kernelDim - 2]) |kernel_row| { //kernel rows
                            kernel_location[kernelDim - 2] = kernel_row;
                            input_location[inDim - 2] = startInputRow + kernel_row;

                            for (0..kernel.shape[kernelDim - 1]) |kernel_cols| { //kernel cols
                                kernel_location[kernelDim - 1] = kernel_cols;
                                input_location[inDim - 1] = startInputCol + kernel_cols;

                                const kernel_value = try kernel.get_at(kernel_location); // kernel_location = [filter_number, channel, kernel_row, kernel_cols]
                                const input_value = input.get_at(input_location) catch |err| { //input_location = [batch, channel, startInputRow + kernel_row, startInputCol + kernel_cols]
                                    // std.debug.print("\n\n  Error!!!  {any}", .{err});
                                    // std.debug.print("\n         get INPUT at:  {any} ", .{input_location});
                                    // std.debug.print("\n         get KERNEL at: {any} ", .{kernel_location});
                                    // std.debug.print("\n         input shape:{any} ", .{input.shape});
                                    // std.debug.print("\n         kernel shape:{any} ", .{kernel.shape});
                                    // std.debug.print("\n         output shape:{any} ", .{output.shape});
                                    // std.debug.print("\n         stride:{any} ", .{stride});

                                    return err;
                                };
                                //std.debug.print("\n         get KERNEL at: {any} value:{any}", .{ kernel_location, kernel_value });
                                //std.debug.print("\n         get INPUT at:  {any} value:{any}", .{ input_location, input_value });

                                sum += kernel_value * input_value;
                            }
                            input_location[inDim - 1] = startInputCol;
                        }
                        input_location[inDim - 2] = startInputRow;
                    }

                    //adding the bias
                    sum += try bias.get_at(bias_location);

                    //set the result in the output tensor
                    //std.debug.print("\n set OUTPUT at:{any} value:{}", .{ output_location, sum });
                    try output.set_at(output_location, sum);
                }
                output_location[outDim - 1] = 0;
            }
        }
    } else {

        // itereate on all the elements at the current input depth
        for (0..input.shape[current_dim]) |i| {
            location[current_dim] = i;

            if (location[current_dim] >= output.shape[current_dim]) {
                std.debug.print("Error: location out of bounds: {d}, shape: {d}\n", .{ location, output.shape });
                return error.IndexOutOfBounds;
            }

            //std.debug.print("multidim_convolution_with_bias: Recursing at dimension {d}, index {d}\n", .{ current_dim, i });

            try multidim_convolution_with_bias(
                inputType,
                outputType,
                input,
                kernel,
                output,
                bias,
                stride,
                current_dim + 1,
                location,
            );
        }
    }
}

/// Convolution tensor with bias
/// TODO: create 2d convolution, atm is 3 or more dimensions
/// TODO: add better check on output size wrt input and kernel
pub fn convolve_tensor_with_bias(
    comptime inputType: anytype,
    comptime outputType: anytype,
    input: *Tensor(inputType), //  shape:[ number of input batches,  number of channels, height, width ]
    kernel: *Tensor(inputType), // shape:[ number of kernel filters, number of channels,  height, width ]
    bias: *Tensor(outputType), //  shape:[ number of baises ]
    stride: []usize, // shape:[row_stride, column_stride]
) !Tensor(outputType) {
    //std.debug.print("CPU_convolve_tensors_with_bias: input shape: {d}, kernel shape: {d}\n", .{ input.shape, kernel.shape });
    const nDimInput = input.shape.len;
    const nDimKernel = kernel.shape.len;
    const nDimOutput = nDimInput;
    const nDimBias = bias.shape.len;

    // std.debug.print("\n -----------------------------------convolve_tensor_with_bias()", .{});
    // std.debug.print("\n input shape:{any} ", .{input.shape});
    // std.debug.print("\n kernel shape:{any} ", .{kernel.shape});
    // std.debug.print("\n bias shape:{any} ", .{bias.shape});
    // std.debug.print("\n stride:{any} ", .{stride});

    //chck on dimensions
    if (nDimKernel > nDimInput) {
        std.debug.print("Error: Kernel size must be smaller or equal to Input size, Kernel size:{}, Input size:{}\n", .{ nDimKernel, nDimInput });
        return TensorMathError.InputTensorDifferentShape;
    }

    //check on input tensor and kernel number of channels, one channel for each filter
    if (input.shape[nDimInput - 3] != kernel.shape[nDimKernel - 3]) {
        std.debug.print("Error: Mismatched channels. Input: {d}, Kernel: {d}\n", .{ input.shape[nDimInput - 3], kernel.shape[nDimKernel - 3] });
        return TensorMathError.InputTensorsWrongShape;
    }

    //check on input tensor and kernel number of rows
    if (kernel.shape[nDimKernel - 2] > input.shape[nDimInput - 2]) {
        std.debug.print("Error: Kernel too big, Input rows: {d}, Kernel rows: {d}\n", .{ input.shape[nDimInput - 2], kernel.shape[nDimKernel - 2] });
        return TensorMathError.InputTensorsWrongShape;
    }

    //check on input tensor and kernel number of cols
    if (kernel.shape[nDimKernel - 1] > input.shape[nDimInput - 1]) {
        std.debug.print("Error: Kernel too big, Input cols: {d}, Kernel cols: {d}\n", .{ input.shape[nDimInput - 2], kernel.shape[nDimKernel - 2] });
        return TensorMathError.InputTensorsWrongShape;
    }

    //check there is one bias for each kernel filter
    if (bias.shape[nDimBias - 1] != kernel.shape[nDimKernel - 4]) {
        std.debug.print("Error: wrong number of biases, # Biases:{}, # Kernel filters:{d}\n", .{ bias.shape.len, kernel.shape[nDimKernel - 3] });
        return TensorMathError.InputTensorsWrongShape;
    }

    //check on the stride size
    if (stride.len != 2) {
        std.debug.print("Error: wrong stride size\n", .{});
        return TensorMathError.WrongStride;
    }
    //check not zero stride
    if (stride[0] == 0 or stride[1] == 0) {
        std.debug.print("Error: stride cannot be zero\n", .{});
        return TensorMathError.WrongStride;
    }

    //output shape
    var out_shape = try pkg_allocator.alloc(usize, nDimOutput);
    defer pkg_allocator.free(out_shape);

    //all the multidimensional batches are the same of the input
    for (0..nDimInput - 3) |i| {
        out_shape[i] = input.shape[i];
    }
    out_shape[nDimOutput - 3] = kernel.shape[nDimKernel - 4]; // n filters
    out_shape[nDimOutput - 2] = (input.shape[nDimInput - 2] - kernel.shape[nDimInput - 2]) / stride[0] + 1; // Height
    out_shape[nDimOutput - 1] = (input.shape[nDimInput - 1] - kernel.shape[nDimInput - 1]) / stride[1] + 1; // Width

    var out_tensor = try Tensor(outputType).fromShape(&pkg_allocator, out_shape);

    //std.debug.print("\n output shape {any}", .{out_shape});

    //initialize the current location to all 0
    //OSS!! the "location" operates on the input, it represents the coordinates in the input space
    const location = try pkg_allocator.alloc(usize, nDimInput);
    defer pkg_allocator.free(location);
    for (location) |*loc| {
        loc.* = 0;
    }

    try multidim_convolution_with_bias(
        inputType,
        outputType,
        input,
        kernel,
        &out_tensor,
        bias,
        stride,
        0,
        location,
    );

    //std.debug.print("Result tensor data: {d}\n", .{out_tensor.data});

    return out_tensor;
}

pub fn convolution_backward_biases(comptime T: type, dValues: *Tensor(T)) !Tensor(T) {
    // Compute gradients with respect to biases by summing over batch, height, and width dimensions
    // Assumes dValues shape: [batch_size, out_channels (aka number of kernel filters ), output_height, output_width]

    // Check that dValues has at least 4 dimensions
    if (dValues.shape.len < 4) return TensorMathError.InputTensorsWrongShape;

    const out_channels = dValues.shape[1];
    var bias_gradients_shape = [_]usize{out_channels};

    // Allocate the bias_gradients tensor
    var bias_gradients = try Tensor(T).fromShape(&pkg_allocator, &bias_gradients_shape);

    const batch_size = dValues.shape[0];
    const output_height = dValues.shape[2];
    const output_width = dValues.shape[3];

    // Sum over batch_size, output_height, output_width dimensions
    for (0..out_channels) |oc| {
        var sum: T = 0;
        for (0..batch_size) |b| {
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const index = [_]usize{ b, oc, h, w };
                    const val = try dValues.get_at(&index);
                    sum += val;
                    //std.debug.print("\n  adding:{} sum:{} index:{any}", .{ val, sum, index });
                }
            }
        }
        // Set the sum in bias_gradients
        try bias_gradients.set_at(&[_]usize{oc}, sum);
    }

    return bias_gradients;
}

pub fn convolution_backward_weights(
    comptime T: type,
    input: *Tensor(T), // Dimensions [batch_size, channels, height, width]
    dValues: *Tensor(T), // Gradient of output
    kernel: *Tensor(T), // Dimensions : [number of kernel filters, number of channels, height, width ]
    stride: [2]usize, // Dimensions [stride_height, stride_width]
) !Tensor(T) { //output: Dimensions [batch_size, num_filters, output_height, output_width]

    //---------------------------- CONSTS ----------------------------
    const num_filters = kernel.shape[0];
    const num_channels = kernel.shape[1];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];

    const batch_size = input.shape[0];
    // const input_channels = input.shape[1];
    const input_height = input.shape[2];
    const input_width = input.shape[3];

    // const dVal_batches = dValues.shape[0];
    // const dVal_channels = dValues.shape[1];
    const dVal_height = dValues.shape[2];
    const dVal_width = dValues.shape[3];

    std.debug.print("\n\n ---------------- convolution_backward_weights() ----------------", .{});
    // std.debug.print("\n input shape:{any} ", .{input.shape});
    // std.debug.print("\n dValues shape:{any} ", .{dValues.shape});
    // std.debug.print("\n kernel shape:{any} ", .{kernel.shape});
    // std.debug.print("\n stride:{any} ", .{stride});

    // ---------------------------- CHECKS ----------------------------

    //---------------- Validate Dimensions

    //Input Tensor Shape:
    if (input.shape.len != 4) {
        std.debug.print("\n\nError: convolution_backward_weights() is only available for 4D input, your input shape:{any}", .{input.shape});
        if (input.shape.len < 4) {
            std.debug.print(", add [1]s dimensions in the front to reach 4D", .{});
        }
        std.debug.print("\n \n ", .{});

        return TensorMathError.InputTensorsWrongShape;
    }

    //dValues Tensor Shape:
    if (dValues.shape.len != 4) {
        std.debug.print("\n\nError: convolution_backward_weights() is only available for 4D dValues, your dValues shape:{any}", .{dValues.shape});
        if (input.shape.len < 4) {
            std.debug.print(", add [1]s dimensions in the front to reach 4D", .{});
        }
        std.debug.print("\n \n ", .{});

        return TensorMathError.InputTensorsWrongShape;
    }

    //kernel Tensor Shape:
    if (kernel.shape.len != 4) {
        std.debug.print("\n\nError: convolution_backward_weights() is only available for 4D kernel, your kernel shape:{any}", .{kernel.shape});
        if (input.shape.len < 4) {
            std.debug.print(", add [1]s dimensions in the front to reach 4D", .{});
        }
        std.debug.print("\n \n ", .{});

        return TensorMathError.InputTensorsWrongShape;
    }

    // ---------------- validate STRIDE:
    //stride values
    if (stride[0] < 1 or stride[1] < 1) {
        std.debug.print("\n\nError: Strides must be grather or equal to 1, your stride is:{any}", .{stride});
        return TensorMathError.InputTensorsWrongShape;
    }
    //stride values
    if (stride[0] < 1 or stride[1] < 1) {
        std.debug.print("\n\nError: Strides must be grather or equal to 1, your stride is:{any}", .{stride});
        return TensorMathError.InputTensorsWrongShape;
    }

    // ---------------- dimension consistency between dValues, input and kernel:
    const expected_dVal_height = (input_height - kernel_height) / stride[0] + 1;
    const expected_dVal_width = (input_width - kernel_width) / stride[1] + 1;
    if (dVal_height != expected_dVal_height or dVal_width != expected_dVal_width) {
        std.debug.print("\n\nError: expected_dVal_height:{} dVal_height:{} expected_dVal_width:{} dVal_width:{}", .{ expected_dVal_height, dVal_height, expected_dVal_width, dVal_width });
        return error.MismatchedDimensions;
    }
    // ---------------- Validate Channels Size:
    if (kernel.shape[1] != input.shape[1]) {
        return error.MismatchedChannels;
    }

    // ---------------- Validate Batch Size:
    if (input.shape[0] != dValues.shape[0]) return error.MismatchedBatchSize;

    //---------------------------- COMPUTE D_KERNEL ----------------------------

    //initialize d_kernel
    var dKernel_shape: [4]usize = [_]usize{ num_filters, num_channels, kernel_height, kernel_width };
    var dKernel = try Tensor(T).fromShape(&pkg_allocator, &dKernel_shape); //fromShape() already initialize to Zero

    //coordinate trackers
    const input_coordinates = try pkg_allocator.alloc(usize, input.shape.len); //coordinates in the input space
    defer pkg_allocator.free(input_coordinates);

    const dVal_coordinates = try pkg_allocator.alloc(usize, dValues.shape.len); //coordinates in the dValues space
    defer pkg_allocator.free(dVal_coordinates);

    const dKernel_coordinates = try pkg_allocator.alloc(usize, dValues.shape.len); //coordinates in the dKernel space
    defer pkg_allocator.free(dKernel_coordinates);

    for (0..batch_size) |batch_index| {
        //set coordinates
        dVal_coordinates[0] = batch_index;
        input_coordinates[0] = batch_index;

        for (0..num_filters) |filter| {
            //set coordinates
            dVal_coordinates[1] = filter;
            dKernel_coordinates[0] = filter;

            for (0..dVal_height) |y| {
                //set coordinates
                dVal_coordinates[2] = y;

                for (0..dVal_width) |x| {
                    //set coordinates
                    dVal_coordinates[3] = x;

                    //get the value
                    //DEBUG std.debug.print("\n         get dVal_value at: {any} ", .{dVal_coordinates});

                    const dVal_value = try dValues.get_at(dVal_coordinates);

                    for (0..num_channels) |channel| {
                        //set coordinates
                        input_coordinates[1] = channel;
                        dKernel_coordinates[1] = channel;

                        for (0..kernel_height) |kernel_y| {
                            //set coordinates
                            dKernel_coordinates[2] = kernel_y;

                            for (0..kernel_width) |kernel_x| {
                                //set coordinates
                                dKernel_coordinates[3] = kernel_x;

                                const in_y = y * stride[0] + kernel_y;
                                const in_x = x * stride[1] + kernel_x;

                                if (in_y < input.shape[2] and in_x < input.shape[3]) {
                                    input_coordinates[2] = in_y;
                                    input_coordinates[3] = in_x;

                                    //DEBUG std.debug.print("\n         get input_value at: {any}", .{input_coordinates});

                                    //input_coordinates at this point= [batch_index, channel, in_y, in_x]
                                    const input_value = try input.get_at(input_coordinates);

                                    var sum = try dKernel.get_at(dKernel_coordinates);

                                    sum += dVal_value * input_value;

                                    //dKernel_coordinates at this point= [filter, channel, kernel_y, kernel_x]
                                    try dKernel.set_at(dKernel_coordinates, sum);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dKernel;
}

/// Computes the backward derivate of the Output with respect to the Input.
/// The operation consist in a full convolution of the flipped Kernel and dValues.
pub fn convolution_backward_input(
    comptime T: type,
    dValues: *Tensor(T),
    kernel: *Tensor(T),
    input: *Tensor(T),
    stride: [2]usize, //Dimensions [stride_height, stride_width]
) !Tensor(T) {
    std.debug.print("\n\n ---------------- convolution_backward_input() ----------------", .{});

    //---------------------------- CONSTS ----------------------------
    const num_filters = kernel.shape[0];
    const num_channels = kernel.shape[1];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];

    const batch_size = input.shape[0];
    // const input_channels = input.shape[1];
    // const input_height = input.shape[2];
    // const input_width = input.shape[3];

    // const dVal_batches = dValues.shape[0];
    // const dVal_channels = dValues.shape[1];
    const dVal_height = dValues.shape[2];
    const dVal_width = dValues.shape[3];

    //---------------------------- COMPUTE D_INPUT ----------------------------

    //initialize d_kernel
    var dInput = try Tensor(T).fromShape(&pkg_allocator, input.shape); //fromShape() already initialize to Zero

    //coordinate trackers
    const dInput_coordinates = try pkg_allocator.alloc(usize, input.shape.len); //coordinates in the input space
    defer pkg_allocator.free(dInput_coordinates);

    const dVal_coordinates = try pkg_allocator.alloc(usize, dValues.shape.len); //coordinates in the dValues space
    defer pkg_allocator.free(dVal_coordinates);

    const kernel_coordinates = try pkg_allocator.alloc(usize, dValues.shape.len); //coordinates in the dKernel space
    defer pkg_allocator.free(kernel_coordinates);

    for (0..batch_size) |batch_index| {
        //set coordinates
        dVal_coordinates[0] = batch_index;
        dInput_coordinates[0] = batch_index;

        for (0..num_filters) |filter| {
            //set coordinates
            dVal_coordinates[1] = filter;
            kernel_coordinates[0] = filter;

            for (0..dVal_height) |y| {
                //set coordinates
                dVal_coordinates[2] = y;

                for (0..dVal_width) |x| {
                    //set coordinates
                    dVal_coordinates[3] = x;

                    //get the value
                    //DEBUG std.debug.print("\n         get dVal_value at: {any} ", .{dVal_coordinates});
                    //dVal_coordinates at this point= [batch_index, filter, y, x]]
                    const dVal_value = try dValues.get_at(dVal_coordinates);

                    for (0..num_channels) |channel| {
                        //set coordinates
                        dInput_coordinates[1] = channel;
                        kernel_coordinates[1] = channel;

                        for (0..kernel_height) |kernel_y| {
                            //set coordinates
                            kernel_coordinates[2] = kernel_y;

                            for (0..kernel_width) |kernel_x| {
                                //set coordinates
                                kernel_coordinates[3] = kernel_x;

                                const in_y = y * stride[0] + kernel_y;
                                const in_x = x * stride[1] + kernel_x;

                                if (in_y < input.shape[2] and in_x < input.shape[3]) {
                                    dInput_coordinates[2] = in_y;
                                    dInput_coordinates[3] = in_x;

                                    //DEBUG std.debug.print("\n         get input_value at: {any}", .{input_coordinates});

                                    //kernel_coordinates at this point= [filter, channel, kernel_y, kkernel_xx]]
                                    const kernel_value = try kernel.get_at(kernel_coordinates);

                                    //dInput_coordinates at this point= [batch_index, channel, in_y, in_x]
                                    var sum = try dInput.get_at(dInput_coordinates);

                                    sum += dVal_value * kernel_value;

                                    //dKernel_coordinates at this point= [filter, channel, kernel_y, kernel_x]
                                    try dInput.set_at(dInput_coordinates, sum);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return dInput;
}

// POOLING -----------------------------------------------------------------------------------------------------------------------
//TODO: add padding
pub fn pool_tensor(
    comptime T: type,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !Tensor(T) {
    const allocator = pkg_allocator;

    // Calcolo output shape
    var outputTensorShape = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(outputTensorShape);
    for (0..input.shape.len - 2) |i| {
        outputTensorShape[i] = input.shape[i];
    }
    const height = input.shape.len - 2;
    const width = input.shape.len - 1;

    outputTensorShape[height] = (input.shape[height] - kernel[0] + 1) / stride[0];
    outputTensorShape[width] = (input.shape[width] - kernel[1] + 1) / stride[1];

    var output = try Tensor(T).fromShape(&allocator, outputTensorShape);

    //const output_rows = outputTensorShape[height];
    //const output_cols = outputTensorShape[width];

    // Il tensor used_windows ha shape [W, input_rows, input_cols]
    // W = output_rows * output_cols
    // Assumiamo che used_windows sia già stato allocato e zero-inizializzato prima di chiamare questa funzione.

    // location
    const location = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(location);
    for (location) |*loc| loc.* = 0;

    try multidim_pooling(
        T,
        input,
        used_windows,
        &output,
        0,
        location,
        kernel,
        stride,
        poolingType,
    );

    return output;
}

pub fn multidim_pooling(
    comptime T: anytype,
    input: *Tensor(T),
    used_windows: *Tensor(u8), // Shape: [W, input_rows, input_cols]
    output: *Tensor(T),
    current_depth: usize,
    location: []usize,
    kernel: []usize,
    stride: []usize,
    poolingType: PoolingType,
) !void {
    if (current_depth == output.shape.len - 2) {
        const allocator = pkg_allocator;

        var temp_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(temp_location);
        var window_location = try allocator.alloc(usize, input.shape.len);
        defer allocator.free(window_location);
        var window_values = try allocator.alloc(T, kernel[0] * kernel[1]);
        defer allocator.free(window_values);

        var output_row_counter: usize = 0;
        var output_col_counter: usize = 0;

        @memcpy(temp_location, location);
        @memcpy(window_location, location);

        temp_location[current_depth] = 0;
        temp_location[current_depth + 1] = 0;

        const input_rows = input.shape[current_depth];
        const input_cols = input.shape[current_depth + 1];

        const output_cols = output.shape[current_depth + 1];

        while (temp_location[current_depth] + kernel[0] <= input.shape[current_depth]) : (temp_location[current_depth] += stride[0]) {
            while (temp_location[current_depth + 1] + kernel[1] <= input.shape[current_depth + 1]) : (temp_location[current_depth + 1] += stride[1]) {
                window_location[current_depth] = temp_location[current_depth];
                window_location[current_depth + 1] = temp_location[current_depth + 1];

                const kernel_rows = kernel[0];
                const kernel_cols = kernel[1];

                const w = output_row_counter * output_cols + output_col_counter;

                for (0..kernel_rows) |i| {
                    window_location[current_depth] += i;
                    for (0..kernel_cols) |j| {
                        window_location[current_depth + 1] += j;
                        window_values[i * kernel_cols + j] = try input.get_at(window_location);
                    }
                    window_location[current_depth + 1] = temp_location[current_depth + 1];
                }

                switch (poolingType) {
                    .Max => {
                        var max = window_values[0];
                        var max_idx: usize = 0;
                        for (0..window_values.len) |i| {
                            if (window_values[i] > max) {
                                max = window_values[i];
                                max_idx = i;
                            }
                        }

                        const row = max_idx / kernel_cols;
                        const col = max_idx % kernel_cols;

                        const max_r = temp_location[current_depth] + row;
                        const max_c = temp_location[current_depth + 1] + col;

                        used_windows.data[w * (input_rows * input_cols) + max_r * input_cols + max_c] = 1;

                        // Set output
                        window_location[current_depth] = output_row_counter;
                        window_location[current_depth + 1] = output_col_counter;
                        try output.set_at(window_location, max);
                    },
                    .Min => {},
                    .Avg => {},
                }

                output_col_counter += 1;
            }
            temp_location[current_depth + 1] = 0;
            output_row_counter += 1;
            output_col_counter = 0;
        }
    } else {
        for (0..output.shape[current_depth]) |element_at_current_depth| {
            location[current_depth] = element_at_current_depth;
            try multidim_pooling(
                T,
                input,
                used_windows,
                output,
                current_depth + 1,
                location,
                kernel,
                stride,
                poolingType,
            );
        }
    }
}
