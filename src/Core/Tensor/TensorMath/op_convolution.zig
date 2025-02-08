const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const dot_product_tensor = @import("op_dot_product.zig").dot_product_tensor;

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
    const kernel_elements = try std.math.mul(usize, try std.math.mul(usize, kernel.shape[1], kernel.shape[2]), kernel.shape[3]);

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

    // Safe copy with overflow checks
    for (0..batch_size) |b| {
        for (0..num_filters) |f| {
            for (0..total_spatial) |i| {
                const src_idx = b * num_filters * total_spatial + f * total_spatial + i;
                const dst_idx = f * batch_size * total_spatial + b * total_spatial + i;
                try dval_reshaped.set_at(&[_]usize{ f, dst_idx % (batch_size * total_spatial) }, dvalues.data[src_idx]);
            }
        }
    }

    var dW = try dot_product_tensor(T, T, &dval_reshaped, &input_col);
    defer dW.deinit();

    const batch_size_f = @as(T, @floatFromInt(batch_size));
    for (dW.data) |*val| {
        val.* = val.* / batch_size_f;
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

    var dval_shape = [_]usize{ try std.math.mul(usize, batch_size, total_spatial), num_filters };
    var dval_reshaped = try Tensor(T).fromShape(&pkg_allocator, &dval_shape);
    defer dval_reshaped.deinit();

    // Safe reshape
    for (0..batch_size) |b| {
        for (0..total_spatial) |i| {
            for (0..num_filters) |f| {
                const src_idx = b * num_filters * total_spatial + f * total_spatial + i;
                const dst_idx = b * total_spatial + i;
                try dval_reshaped.set_at(&[_]usize{ dst_idx, f }, dvalues.data[src_idx]);
            }
        }
    }

    const kernel_spatial = kernel_height * kernel_width;
    var transposed_shape = [_]usize{ num_filters, try std.math.mul(usize, channels, kernel_spatial) };
    var kernel_transposed = try Tensor(T).fromShape(&pkg_allocator, &transposed_shape);
    defer kernel_transposed.deinit();

    // Safe transpose
    for (0..num_filters) |f| {
        for (0..channels) |c| {
            for (0..kernel_spatial) |k| {
                const src_idx = try std.math.add(usize, try std.math.mul(usize, f, try std.math.mul(usize, channels, kernel_spatial)), try std.math.add(usize, try std.math.mul(usize, c, kernel_spatial), k));
                const dst_idx = try std.math.add(usize, try std.math.mul(usize, c, kernel_spatial), k);
                try kernel_transposed.set_at(&[_]usize{ f, dst_idx }, kernel.data[src_idx]);
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

    var row: usize = 0;
    while (row < batch_size * out_height * out_width) : (row += 1) {
        const b = row / (out_height * out_width);
        const oh = (row % (out_height * out_width)) / out_width;
        const ow = row % out_width;

        var col: usize = 0;
        while (col < channels * kernel_h * kernel_w) : (col += 1) {
            const c = col / (kernel_h * kernel_w);
            const kh = (col % (kernel_h * kernel_w)) / kernel_w;
            const kw = col % kernel_w;

            const h_offset = try std.math.add(usize, try std.math.mul(usize, oh, stride_h), kh);
            const w_offset = try std.math.add(usize, try std.math.mul(usize, ow, stride_w), kw);

            const input_idx = try std.math.add(usize, try std.math.mul(usize, b, channels * height * width), try std.math.add(usize, try std.math.mul(usize, c, height * width), try std.math.add(usize, try std.math.mul(usize, h_offset, width), w_offset)));

            try output.set_at(&[_]usize{ row, col }, input.data[input_idx]);
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

    var row: usize = 0;
    while (row < batch_size * out_height * out_width) : (row += 1) {
        const b = row / (out_height * out_width);
        const oh = (row % (out_height * out_width)) / out_width;
        const ow = row % out_width;

        var col: usize = 0;
        while (col < channels * kernel_h * kernel_w) : (col += 1) {
            const c = col / (kernel_h * kernel_w);
            const kh = (col % (kernel_h * kernel_w)) / kernel_w;
            const kw = col % kernel_w;

            const h_offset = oh * stride_h + kh;
            const w_offset = ow * stride_w + kw;

            const output_idx = b * channels * height * width + c * height * width + h_offset * width + w_offset;

            const val = try col_matrix.get_at(&[_]usize{ row, col });
            const current = output.data[output_idx];
            output.data[output_idx] = current + val;
        }
    }
}
