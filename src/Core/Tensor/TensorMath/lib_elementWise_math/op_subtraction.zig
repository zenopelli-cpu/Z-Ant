const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
/// Performs element-wise binary subtraction with Numpy-style broadcasting support
pub fn sub_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    if (@TypeOf(outputType) == @TypeOf(inputType)) {
        // If input and output are same type, no check needed
    } else {
        if (@bitSizeOf(outputType) <= 16) { //quantized
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { //non-quant
            if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    // Handle broadcasting
    const rank1 = t1.shape.len;
    const rank2 = t2.shape.len;
    const max_rank = @max(rank1, rank2);

    // Create output tensor with broadcasted shape
    var out_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(out_shape);

    // Pad shapes with 1s for broadcasting
    var shape1 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(shape1);
    var shape2 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(shape2);

    // Initialize with 1s
    @memset(shape1, 1);
    @memset(shape2, 1);

    // Copy original shapes from right to left
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }
    i = 0;
    while (i < rank2) : (i += 1) {
        shape2[max_rank - rank2 + i] = t2.shape[i];
    }

    // Calculate broadcasted shape
    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Calculate total size
    var total_size: usize = 1;
    for (0..max_rank) |dim| {
        total_size *= out_shape[dim];
    }

    // Create output tensor
    var out_tensor = try Tensor(outputType).fromShape(&pkg_allocator, out_shape);
    errdefer out_tensor.deinit();
    pkg_allocator.free(out_shape); // Free out_shape after creating tensor

    try lean_sub_tensors(inputType, outputType, t1, t2, &out_tensor);

    return out_tensor;
}

pub inline fn lean_sub_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    const rank1 = t1.shape.len;
    const rank2 = t2.shape.len;
    const max_rank = @max(rank1, rank2);

    // Fast path: identical shapes, no broadcasting needed
    if (rank1 == rank2 and std.mem.eql(usize, t1.shape, t2.shape)) {
        // Use SIMD for larger sizes when possible
        const vector_len: usize = std.simd.suggestVectorLength(inputType) orelse 4;
        const Vec = @Vector(vector_len, inputType);

        if (t1.size >= vector_len) {
            // Process data in SIMD-sized chunks
            const aligned_len = (t1.size / vector_len) * vector_len;
            var i: usize = 0;

            // Main SIMD loop
            while (i < aligned_len) : (i += vector_len) {
                const v1: Vec = t1.data[i..][0..vector_len].*;
                const v2: Vec = t2.data[i..][0..vector_len].*;
                const result = v1 - v2;

                // Store results
                for (0..vector_len) |j| {
                    outputTensor.data[i + j] = @as(outputType, result[j]);
                }
            }

            // Handle remaining elements
            while (i < t1.size) : (i += 1) {
                outputTensor.data[i] = @as(outputType, t1.data[i] - t2.data[i]);
            }
            return;
        }
    }

    // Broadcasting path
    var stack_shape1: [8]usize = [_]usize{1} ** 8;
    var stack_shape2: [8]usize = [_]usize{1} ** 8;
    var stack_strides1: [8]usize = undefined;
    var stack_strides2: [8]usize = undefined;
    var stack_out_strides: [8]usize = undefined;

    // Use stack for common ranks, heap for larger
    const shape1 = if (max_rank <= 8) stack_shape1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const shape2 = if (max_rank <= 8) stack_shape2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides1 = if (max_rank <= 8) stack_strides1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides2 = if (max_rank <= 8) stack_strides2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const out_strides = if (max_rank <= 8) stack_out_strides[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);

    if (max_rank > 8) {
        defer {
            pkg_allocator.free(shape1);
            pkg_allocator.free(shape2);
            pkg_allocator.free(strides1);
            pkg_allocator.free(strides2);
            pkg_allocator.free(out_strides);
        }
    }

    // Prepare shapes for broadcasting
    @memset(shape1, 1);
    @memset(shape2, 1);

    // Copy shapes from right to left
    for (0..rank1) |i| {
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }
    for (0..rank2) |i| {
        shape2[max_rank - rank2 + i] = t2.shape[i];
    }

    // Calculate strides for efficient access
    var stride: usize = 1;
    var i = max_rank;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        strides1[i] = if (shape1[i] > 1) stride else 0;
        strides2[i] = if (shape2[i] > 1) stride else 0;
        const offset = max_rank -| i; // saturating subtraction
        const shape_idx = if (offset < outputTensor.shape.len)
            outputTensor.shape.len -| offset -| 1
        else
            0;
        stride *= if (shape_idx < outputTensor.shape.len) outputTensor.shape[shape_idx] else 1;
    }

    // Process elements with broadcasting
    const vector_len: usize = std.simd.suggestVectorLength(inputType) orelse 4;
    const Vec = @Vector(vector_len, inputType);
    var contiguous_run: usize = 0;
    var last_idx1: usize = 0;
    var last_idx2: usize = 0;

    i = 0;
    while (i < outputTensor.size) : (i += 1) {
        // Calculate input indices for broadcasting
        var idx1: usize = 0;
        var idx2: usize = 0;
        var temp: usize = i;

        // Calculate indices for both tensors
        var dim: usize = 0;
        while (dim < max_rank) : (dim += 1) {
            const current_stride = out_strides[dim];
            if (current_stride > 0) {
                const pos = temp / current_stride;
                temp = temp - (pos * current_stride);

                if (shape1[dim] > 1) {
                    idx1 += pos * strides1[dim];
                }
                if (shape2[dim] > 1) {
                    const t2_pos = if (shape2[dim] > 1) pos % shape2[dim] else pos;
                    idx2 += t2_pos * strides2[dim];
                }
            }
        }

        // Check if we can use SIMD for a contiguous run
        if (i > 0 and idx1 == last_idx1 + 1 and idx2 == last_idx2 + 1) {
            contiguous_run += 1;
            if (contiguous_run >= vector_len) {
                const start = i - vector_len + 1;
                const v1: Vec = t1.data[idx1 - vector_len + 1 ..][0..vector_len].*;
                const v2: Vec = t2.data[idx2 - vector_len + 1 ..][0..vector_len].*;
                const result = v1 - v2;

                inline for (0..vector_len) |j| {
                    outputTensor.data[start + j] = @as(outputType, result[j]);
                }
                contiguous_run = 0;
                continue;
            }
        } else {
            contiguous_run = 0;
        }

        // Scalar operation for non-contiguous or remaining elements
        outputTensor.data[i] = @as(outputType, t1.data[idx1] - t2.data[idx2]);
        last_idx1 = idx1;
        last_idx2 = idx2;
    }
}
