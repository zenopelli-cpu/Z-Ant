const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const dot_product = @import("op_dot_product.zig");
const dot_product_tensor = dot_product.dot_product_tensor;

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
/// Convolution tensor with bias
/// TODO: create 2d convolution, atm is 3 or more dimensions
/// TODO: add better check on output size wrt input and kernel
pub fn convolve_tensor_with_bias(
    comptime T: type,
    input: *Tensor(T),
    kernel: *const Tensor(T),
    bias: *const Tensor(T),
    stride: []const usize, // shape:[row_stride, column_stride]
) !Tensor(T) {
    const nDimInput = input.shape.len;
    const nDimKernel = kernel.shape.len;
    const nDimBias = bias.shape.len;

    if (nDimInput != 4 or nDimKernel != 4) {
        return TensorMathError.InvalidDimensions;
    }

    if (nDimKernel > nDimInput) {
        return TensorMathError.InputTensorDifferentShape;
    }

    if (input.shape[nDimInput - 3] != kernel.shape[nDimKernel - 3]) {
        return TensorMathError.InputTensorsWrongShape;
    }

    if (kernel.shape[nDimKernel - 2] > input.shape[nDimInput - 2] or
        kernel.shape[nDimKernel - 1] > input.shape[nDimInput - 1])
    {
        return TensorMathError.InputTensorsWrongShape;
    }

    if (bias.shape[nDimBias - 1] != kernel.shape[nDimKernel - 4]) {
        return TensorMathError.InputTensorsWrongShape;
    }

    if (stride.len != 2 or stride[0] == 0 or stride[1] == 0) {
        return TensorMathError.WrongStride;
    }

    // Check if dimensions are valid for stride
    const out_height = try std.math.divExact(usize, input.shape[2] - kernel.shape[2], stride[0]) + 1;
    const out_width = try std.math.divExact(usize, input.shape[3] - kernel.shape[3], stride[1]) + 1;

    const kernel_size = [2]usize{ kernel.shape[2], kernel.shape[3] };
    const stride_size = [2]usize{ stride[0], stride[1] };

    var input_col = try im2col(T, input, kernel_size, stride_size);
    defer input_col.deinit();

    const num_filters = kernel.shape[0];
    const kernel_elements = try std.math.mul(usize, kernel.shape[1], try std.math.mul(usize, kernel.shape[2], kernel.shape[3]));

    var kernel_matrix_shape = [_]usize{ kernel_elements, num_filters };
    var kernel_matrix = try Tensor(T).fromShape(&pkg_allocator, &kernel_matrix_shape);
    defer kernel_matrix.deinit();

    // Safely copy kernel data
    for (0..num_filters) |f| {
        for (0..kernel_elements) |i| {
            const idx = try std.math.mul(usize, f, kernel_elements);
            try kernel_matrix.set_at(&[_]usize{ i, f }, kernel.data[idx + i]);
        }
    }

    const batch_size = input.shape[0];
    var output_shape = [_]usize{ batch_size, num_filters, out_height, out_width };
    var output = try Tensor(T).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    var result = try dot_product_tensor(T, T, &input_col, &kernel_matrix);
    defer result.deinit();

    // Safe copy with direct floating point addition
    var idx: usize = 0;
    for (0..batch_size) |b| {
        for (0..out_height) |h| {
            for (0..out_width) |w| {
                for (0..num_filters) |f| {
                    const val = try result.get_at(&[_]usize{ idx, f });
                    const bias_val = bias.data[f];
                    try output.set_at(&[_]usize{ b, f, h, w }, val + bias_val); // Direct floating point addition
                }
                idx += 1;
            }
        }
    }

    return output;
}

pub fn get_convolution_output_shape(input_shape: []const usize, kernel_shape: []const usize, stride: []const usize) ![4]usize {
    if (input_shape.len != 4 or kernel_shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    if (stride.len != 2 or stride[0] == 0 or stride[1] == 0) {
        return TensorMathError.WrongStride;
    }

    const batch_size = input_shape[0];
    const num_filters = kernel_shape[0];
    const out_height = try std.math.divExact(usize, input_shape[2] - kernel_shape[2], stride[0]) + 1;
    const out_width = try std.math.divExact(usize, input_shape[3] - kernel_shape[3], stride[1]) + 1;

    return [4]usize{ batch_size, num_filters, out_height, out_width };
}

pub fn convolution_backward_biases(comptime T: type, dValues: *Tensor(T)) !Tensor(T) {
    if (dValues.shape.len != 4) return TensorMathError.InvalidDimensions;

    const out_channels = dValues.shape[1];
    var bias_gradients_shape = [_]usize{out_channels};

    var bias_gradients = try Tensor(T).fromShape(&pkg_allocator, &bias_gradients_shape);
    errdefer bias_gradients.deinit();
    try bias_gradients.set(0, 0);

    const batch_size = dValues.shape[0];
    const output_height = dValues.shape[2];
    const output_width = dValues.shape[3];

    for (0..out_channels) |oc| {
        var sum: T = 0;
        for (0..batch_size) |b| {
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const val = try dValues.get_at(&[_]usize{ b, oc, h, w });
                    sum += val; // Direct addition for floating point
                }
            }
        }
        try bias_gradients.set_at(&[_]usize{oc}, sum);
    }

    return bias_gradients;
}

pub fn convolution_backward_weights(comptime T: type, input: *Tensor(T), dvalues: *Tensor(T), kernel_shape: []const usize, stride: [2]usize) !Tensor(T) {
    if (kernel_shape.len != 4) return TensorMathError.InvalidDimensions;

    // Create mutable copy of kernel_shape
    var mutable_kernel_shape: [4]usize = undefined;
    @memcpy(&mutable_kernel_shape, kernel_shape);

    const batch_size = input.shape[0];
    const num_filters = kernel_shape[0];
    const kernel_height = kernel_shape[2];
    const kernel_width = kernel_shape[3];

    const kernel_size = [2]usize{ kernel_height, kernel_width };
    var input_col = try im2col(T, input, kernel_size, stride);
    defer input_col.deinit();

    const out_height = dvalues.shape[2];
    const out_width = dvalues.shape[3];
    const total_spatial = out_height * out_width;

    var dval_shape = [_]usize{ num_filters, batch_size * total_spatial };
    var dval_reshaped = try Tensor(T).fromShape(&pkg_allocator, &dval_shape);
    defer dval_reshaped.deinit();

    // Pre-compute strides for faster access
    const out_channel_stride = out_height * out_width;
    const out_batch_stride = num_filters * out_channel_stride;

    // Use SIMD for copying when possible
    const Vector = @Vector(4, T);
    const can_use_simd = total_spatial >= 4;

    // Safe copy with optimized memory access
    for (0..batch_size) |b| {
        const batch_offset = b * out_batch_stride;
        const dst_batch_offset = b * total_spatial;

        for (0..num_filters) |f| {
            const src_offset = batch_offset + f * out_channel_stride;
            const dst_offset = f * batch_size * total_spatial + dst_batch_offset;

            if (can_use_simd) {
                var i: usize = 0;
                while (i + 4 <= total_spatial) : (i += 4) {
                    const vec = Vector{
                        dvalues.data[src_offset + i],
                        dvalues.data[src_offset + i + 1],
                        dvalues.data[src_offset + i + 2],
                        dvalues.data[src_offset + i + 3],
                    };

                    dval_reshaped.data[dst_offset + i] = vec[0];
                    dval_reshaped.data[dst_offset + i + 1] = vec[1];
                    dval_reshaped.data[dst_offset + i + 2] = vec[2];
                    dval_reshaped.data[dst_offset + i + 3] = vec[3];
                }

                // Handle remaining elements
                var i_rem = total_spatial - (total_spatial % 4);
                while (i_rem < total_spatial) : (i_rem += 1) {
                    dval_reshaped.data[dst_offset + i_rem] = dvalues.data[src_offset + i_rem];
                }
            } else {
                for (0..total_spatial) |i| {
                    dval_reshaped.data[dst_offset + i] = dvalues.data[src_offset + i];
                }
            }
        }
    }

    var dW = try dot_product_tensor(T, T, &dval_reshaped, &input_col);
    defer dW.deinit();

    // Use SIMD for division
    const batch_size_f = @as(T, @floatFromInt(batch_size));
    if (can_use_simd) {
        const batch_size_vec = Vector{ batch_size_f, batch_size_f, batch_size_f, batch_size_f };
        var i: usize = 0;
        while (i + 4 <= dW.data.len) : (i += 4) {
            const vec = Vector{
                dW.data[i],
                dW.data[i + 1],
                dW.data[i + 2],
                dW.data[i + 3],
            };
            const result = vec / batch_size_vec;
            dW.data[i] = result[0];
            dW.data[i + 1] = result[1];
            dW.data[i + 2] = result[2];
            dW.data[i + 3] = result[3];
        }

        // Handle remaining elements
        var i_rem = dW.data.len - (dW.data.len % 4);
        while (i_rem < dW.data.len) : (i_rem += 1) {
            dW.data[i_rem] = dW.data[i_rem] / batch_size_f;
        }
    } else {
        for (dW.data) |*val| {
            val.* = val.* / batch_size_f;
        }
    }

    var dW_reshaped = try Tensor(T).fromShape(&pkg_allocator, &mutable_kernel_shape);
    errdefer dW_reshaped.deinit();

    @memcpy(dW_reshaped.data, dW.data);

    return dW_reshaped;
}

pub fn convolution_backward_input(comptime T: type, dvalues: *const Tensor(T), kernel: *const Tensor(T), input_shape: []const usize, stride: [2]usize) !Tensor(T) {
    if (input_shape.len != 4) return TensorMathError.InvalidDimensions;

    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const kernel_height = kernel.shape[2];
    const kernel_width = kernel.shape[3];
    const num_filters = kernel.shape[0];

    const out_height = dvalues.shape[2];
    const out_width = dvalues.shape[3];
    const total_spatial = out_height * out_width;

    const total_batch_spatial = try std.math.mul(usize, batch_size, total_spatial);
    var dval_shape = [_]usize{ total_batch_spatial, num_filters };
    var dval_reshaped = try Tensor(T).fromShape(&pkg_allocator, &dval_shape);
    defer dval_reshaped.deinit();

    // Pre-compute strides for faster access
    const dval_channel_stride = out_height * out_width;
    const dval_batch_stride = num_filters * dval_channel_stride;

    // Use SIMD for copying when possible
    const Vector = @Vector(4, T);
    const can_use_simd = num_filters >= 4;

    // Safe reshape with optimized memory access
    for (0..batch_size) |b| {
        const batch_offset = b * dval_batch_stride;
        const dst_batch_offset = b * total_spatial;

        for (0..total_spatial) |i| {
            const src_base = batch_offset + i % dval_channel_stride;
            const dst_base = dst_batch_offset + i;

            if (can_use_simd) {
                var f: usize = 0;
                while (f + 4 <= num_filters) : (f += 4) {
                    const vec = Vector{
                        dvalues.data[src_base + f * dval_channel_stride],
                        dvalues.data[src_base + (f + 1) * dval_channel_stride],
                        dvalues.data[src_base + (f + 2) * dval_channel_stride],
                        dvalues.data[src_base + (f + 3) * dval_channel_stride],
                    };

                    dval_reshaped.data[dst_base * num_filters + f] = vec[0];
                    dval_reshaped.data[dst_base * num_filters + f + 1] = vec[1];
                    dval_reshaped.data[dst_base * num_filters + f + 2] = vec[2];
                    dval_reshaped.data[dst_base * num_filters + f + 3] = vec[3];
                }

                // Handle remaining filters
                var f_rem = num_filters - (num_filters % 4);
                while (f_rem < num_filters) : (f_rem += 1) {
                    dval_reshaped.data[dst_base * num_filters + f_rem] = dvalues.data[src_base + f_rem * dval_channel_stride];
                }
            } else {
                for (0..num_filters) |f| {
                    dval_reshaped.data[dst_base * num_filters + f] = dvalues.data[src_base + f * dval_channel_stride];
                }
            }
        }
    }

    const kernel_spatial = kernel_height * kernel_width;
    var transposed_shape = [_]usize{ num_filters, try std.math.mul(usize, channels, kernel_spatial) };
    var kernel_transposed = try Tensor(T).fromShape(&pkg_allocator, &transposed_shape);
    defer kernel_transposed.deinit();

    // Pre-compute strides for kernel transpose
    const kernel_filter_stride = channels * kernel_spatial;

    // Safe transpose with optimized memory access
    for (0..num_filters) |f| {
        const filter_offset = f * kernel_filter_stride;
        const src_filter_offset = f * channels * kernel_spatial;

        for (0..channels) |c| {
            const channel_offset = filter_offset + c * kernel_spatial;
            const src_channel_offset = src_filter_offset + c * kernel_spatial;

            if (can_use_simd and kernel_spatial >= 4) {
                var k: usize = 0;
                while (k + 4 <= kernel_spatial) : (k += 4) {
                    const vec = Vector{
                        kernel.data[src_channel_offset + k],
                        kernel.data[src_channel_offset + k + 1],
                        kernel.data[src_channel_offset + k + 2],
                        kernel.data[src_channel_offset + k + 3],
                    };

                    kernel_transposed.data[channel_offset + k] = vec[0];
                    kernel_transposed.data[channel_offset + k + 1] = vec[1];
                    kernel_transposed.data[channel_offset + k + 2] = vec[2];
                    kernel_transposed.data[channel_offset + k + 3] = vec[3];
                }

                // Handle remaining elements
                var k_rem = kernel_spatial - (kernel_spatial % 4);
                while (k_rem < kernel_spatial) : (k_rem += 1) {
                    kernel_transposed.data[channel_offset + k_rem] = kernel.data[src_channel_offset + k_rem];
                }
            } else {
                for (0..kernel_spatial) |k| {
                    kernel_transposed.data[channel_offset + k] = kernel.data[src_channel_offset + k];
                }
            }
        }
    }

    var dX_col = try dot_product_tensor(T, T, &dval_reshaped, &kernel_transposed);
    defer dX_col.deinit();

    const kernel_size = [2]usize{ kernel_height, kernel_width };
    return try col2im(T, &dX_col, input_shape, kernel_size, stride);
}

// --------------------------------------------------
// --------------------- im2col ---------------------
// --------------------------------------------------
// --------- standard im2col
pub fn im2col(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize) !Tensor(T) {
    if (input.shape.len != 4) {
        return TensorMathError.InputTensorsWrongShape;
    }

    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];

    if (height < kernel_h or width < kernel_w) {
        return TensorMathError.InvalidDimensions;
    }

    const out_height = try std.math.divExact(usize, height - kernel_h, stride_h) + 1;
    const out_width = try std.math.divExact(usize, width - kernel_w, stride_w) + 1;

    const rows = try std.math.mul(usize, batch_size, try std.math.mul(usize, out_height, out_width));
    const cols = try std.math.mul(usize, channels, try std.math.mul(usize, kernel_h, kernel_w));

    var col_shape = [_]usize{ rows, cols };
    var col_matrix = try Tensor(T).fromShape(&pkg_allocator, &col_shape);
    errdefer col_matrix.deinit();

    try lean_im2col(T, input, kernel, stride, &col_matrix);

    return col_matrix;
}
// --------- lean im2col
pub inline fn lean_im2col(comptime T: type, input: *Tensor(T), kernel: [2]usize, stride: [2]usize, output: *Tensor(T)) !void {
    const batch_size = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];

    const out_height = try std.math.divExact(usize, height - kernel_h, stride_h) + 1;
    const out_width = try std.math.divExact(usize, width - kernel_w, stride_w) + 1;

    // Pre-compute strides for faster access
    const input_channel_stride = height * width;
    const input_batch_stride = channels * input_channel_stride;
    const kernel_size = kernel_h * kernel_w;
    const output_col_stride = channels * kernel_size;

    // Use SIMD for copying when possible
    const Vector = @Vector(4, T);
    const can_use_simd = kernel_w >= 4;

    var row: usize = 0;
    while (row < batch_size * out_height * out_width) : (row += 1) {
        const b = row / (out_height * out_width);
        const oh = (row % (out_height * out_width)) / out_width;
        const ow = row % out_width;

        // Pre-compute base indices
        const batch_offset = b * input_batch_stride;
        const h_offset = oh * stride_h;
        const w_offset = ow * stride_w;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const channel_offset = batch_offset + c * input_channel_stride;
            const output_offset = row * output_col_stride + c * kernel_size;

            var kh: usize = 0;
            while (kh < kernel_h) : (kh += 1) {
                const h_idx = h_offset + kh;
                const row_offset = channel_offset + h_idx * width;

                if (can_use_simd) {
                    var kw: usize = 0;
                    while (kw + 4 <= kernel_w) : (kw += 4) {
                        const w_idx = w_offset + kw;
                        const input_idx = row_offset + w_idx;
                        const out_idx = output_offset + kh * kernel_w + kw;

                        const vec = Vector{
                            input.data[input_idx],
                            input.data[input_idx + 1],
                            input.data[input_idx + 2],
                            input.data[input_idx + 3],
                        };

                        output.data[out_idx] = vec[0];
                        output.data[out_idx + 1] = vec[1];
                        output.data[out_idx + 2] = vec[2];
                        output.data[out_idx + 3] = vec[3];
                    }

                    // Handle remaining elements
                    var kw_rem = kernel_w - (kernel_w % 4);
                    while (kw_rem < kernel_w) : (kw_rem += 1) {
                        const w_idx = w_offset + kw_rem;
                        const input_idx = row_offset + w_idx;
                        const out_idx = output_offset + kh * kernel_w + kw_rem;
                        output.data[out_idx] = input.data[input_idx];
                    }
                } else {
                    var kw: usize = 0;
                    while (kw < kernel_w) : (kw += 1) {
                        const w_idx = w_offset + kw;
                        const input_idx = row_offset + w_idx;
                        const out_idx = output_offset + kh * kernel_w + kw;
                        output.data[out_idx] = input.data[input_idx];
                    }
                }
            }
        }
    }
}

/// Converts a 2D matrix back to a 4D tensor using col2im algorithm
/// Input shape: [batch_size * out_height * out_width, channels * kernel_height * kernel_width]
/// Output shape: [batch_size, channels, height, width]
pub fn col2im(comptime T: type, col_matrix: *Tensor(T), output_shape: []const usize, kernel: [2]usize, stride: [2]usize) !Tensor(T) {
    if (output_shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    var shape: [4]usize = std.mem.zeroes([4]usize);
    @memcpy(&shape, output_shape[0..4]);

    var output = try Tensor(T).fromShape(&pkg_allocator, &shape);
    errdefer output.deinit();
    try output.set(0, 0);

    try lean_col2im(T, col_matrix, output_shape, kernel, stride, &output);

    return output;
}

pub inline fn lean_col2im(comptime T: type, col_matrix: *Tensor(T), output_shape: []const usize, kernel: [2]usize, stride: [2]usize, output: *Tensor(T)) !void {
    const batch_size = output_shape[0];
    const channels = output_shape[1];
    const height = output_shape[2];
    const width = output_shape[3];

    const kernel_h = kernel[0];
    const kernel_w = kernel[1];
    const stride_h = stride[0];
    const stride_w = stride[1];

    const out_height = try std.math.divExact(usize, height - kernel_h, stride_h) + 1;
    const out_width = try std.math.divExact(usize, width - kernel_w, stride_w) + 1;

    // Pre-compute strides for faster access
    const output_channel_stride = height * width;
    const output_batch_stride = channels * output_channel_stride;
    const kernel_size = kernel_h * kernel_w;
    const col_channel_stride = kernel_size;
    const col_batch_stride = channels * col_channel_stride;

    // Use SIMD for accumulation when possible
    const Vector = @Vector(4, T);
    const can_use_simd = kernel_w >= 4;

    var row: usize = 0;
    while (row < batch_size * out_height * out_width) : (row += 1) {
        const b = row / (out_height * out_width);
        const oh = (row % (out_height * out_width)) / out_width;
        const ow = row % out_width;

        // Pre-compute base indices
        const batch_offset = b * output_batch_stride;
        const h_offset = oh * stride_h;
        const w_offset = ow * stride_w;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const channel_offset = batch_offset + c * output_channel_stride;
            const col_offset = row * col_batch_stride + c * col_channel_stride;

            var kh: usize = 0;
            while (kh < kernel_h) : (kh += 1) {
                const h_idx = h_offset + kh;
                if (h_idx >= height) continue;
                const row_offset = channel_offset + h_idx * width;

                if (can_use_simd) {
                    var kw: usize = 0;
                    while (kw + 4 <= kernel_w) : (kw += 4) {
                        const w_idx = w_offset + kw;
                        if (w_idx + 3 >= width) break;

                        const output_idx = row_offset + w_idx;
                        const col_idx = col_offset + kh * kernel_w + kw;

                        // Load current values
                        const curr_vec = Vector{
                            output.data[output_idx],
                            output.data[output_idx + 1],
                            output.data[output_idx + 2],
                            output.data[output_idx + 3],
                        };

                        // Load values to add
                        const add_vec = Vector{
                            col_matrix.data[col_idx],
                            col_matrix.data[col_idx + 1],
                            col_matrix.data[col_idx + 2],
                            col_matrix.data[col_idx + 3],
                        };

                        // Add and store
                        const result = curr_vec + add_vec;
                        output.data[output_idx] = result[0];
                        output.data[output_idx + 1] = result[1];
                        output.data[output_idx + 2] = result[2];
                        output.data[output_idx + 3] = result[3];
                    }

                    // Handle remaining elements
                    var kw_rem = kernel_w - (kernel_w % 4);
                    while (kw_rem < kernel_w) : (kw_rem += 1) {
                        const w_idx = w_offset + kw_rem;
                        if (w_idx >= width) continue;

                        const output_idx = row_offset + w_idx;
                        const col_idx = col_offset + kh * kernel_w + kw_rem;
                        output.data[output_idx] += col_matrix.data[col_idx];
                    }
                } else {
                    var kw: usize = 0;
                    while (kw < kernel_w) : (kw += 1) {
                        const w_idx = w_offset + kw;
                        if (w_idx >= width) continue;

                        const output_idx = row_offset + w_idx;
                        const col_idx = col_offset + kh * kernel_w + kw;
                        output.data[output_idx] += col_matrix.data[col_idx];
                    }
                }
            }
        }
    }
}
