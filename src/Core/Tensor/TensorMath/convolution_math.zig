const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;

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
    const dVal_height = dValues.shape[2];
    const dVal_width = dValues.shape[3];

    //---------------------------- COMPUTE D_KERNEL ----------------------------

    //initialize d_kernel
    var dKernel_shape: [4]usize = [_]usize{ num_filters, num_channels, kernel_height, kernel_width };
    var dKernel = try Tensor(T).fromShape(&pkg_allocator, &dKernel_shape); //fromShape() already initialize to Zero

    //coordinate trackers
    const input_coordinates = try pkg_allocator.alloc(usize, input.shape.len); //coordinates in the input space
    defer pkg_allocator.free(input_coordinates);

    const dVal_coordinates = try pkg_allocator.alloc(usize, dValues.shape.len); //coordinates in the dValues space
    defer pkg_allocator.free(dVal_coordinates);

    const kernel_coordinates = try pkg_allocator.alloc(usize, kernel.shape.len); //coordinates in the kernel space
    defer pkg_allocator.free(kernel_coordinates);

    for (0..batch_size) |batch_index| {
        //set coordinates
        dVal_coordinates[0] = batch_index;
        input_coordinates[0] = batch_index;

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

                    const dVal_value = try dValues.get_at(dVal_coordinates);

                    for (0..num_channels) |channel| {
                        //set coordinates
                        input_coordinates[1] = channel;
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
                                    input_coordinates[2] = in_y;
                                    input_coordinates[3] = in_x;

                                    const input_value = try input.get_at(input_coordinates);
                                    var sum = try dKernel.get_at(kernel_coordinates);
                                    sum += dVal_value * input_value;
                                    try dKernel.set_at(kernel_coordinates, sum);
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
    stride: [2]usize,
) !Tensor(T) {
    var dInput = try Tensor(T).fromShape(&pkg_allocator, input.shape);

    const input_coordinates = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(input_coordinates);

    const dVal_coordinates = try pkg_allocator.alloc(usize, dValues.shape.len);
    defer pkg_allocator.free(dVal_coordinates);

    const kernel_coordinates = try pkg_allocator.alloc(usize, kernel.shape.len);
    defer pkg_allocator.free(kernel_coordinates);

    const batch_size = input.shape[0];
    const num_filters = kernel.shape[0];
    const num_channels = kernel.shape[1];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];
    const dVal_height = dValues.shape[2];
    const dVal_width = dValues.shape[3];

    for (0..batch_size) |batch_index| {
        dVal_coordinates[0] = batch_index;
        input_coordinates[0] = batch_index;

        for (0..num_filters) |filter| {
            dVal_coordinates[1] = filter;
            kernel_coordinates[0] = filter;

            for (0..dVal_height) |y| {
                dVal_coordinates[2] = y;

                for (0..dVal_width) |x| {
                    dVal_coordinates[3] = x;
                    const dVal_value = try dValues.get_at(dVal_coordinates);

                    for (0..num_channels) |channel| {
                        input_coordinates[1] = channel;
                        kernel_coordinates[1] = channel;

                        for (0..kernel_height) |kernel_y| {
                            kernel_coordinates[2] = kernel_y;

                            for (0..kernel_width) |kernel_x| {
                                kernel_coordinates[3] = kernel_x;

                                const in_y = y * stride[0] + kernel_y;
                                const in_x = x * stride[1] + kernel_x;

                                if (in_y < input.shape[2] and in_x < input.shape[3]) {
                                    input_coordinates[2] = in_y;
                                    input_coordinates[3] = in_x;

                                    const kernel_value = try kernel.get_at(kernel_coordinates);
                                    var sum = try dInput.get_at(input_coordinates);
                                    sum += dVal_value * kernel_value;
                                    try dInput.set_at(input_coordinates, sum);
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
