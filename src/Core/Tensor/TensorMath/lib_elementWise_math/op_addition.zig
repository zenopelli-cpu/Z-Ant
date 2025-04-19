const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

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

    // Instead of using threads, just do it directly
    var index: usize = 0;
    while (index < tensor.size) : (index += len) {
        for (0..len) |i| {
            tensor.data[index + i] += bias.data[i];
        }
    }
}

pub fn sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *const Tensor(inputType), t2: *const Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    if (t1.size != t2.size) return TensorMathError.InputTensorDifferentSize;

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    }

    // Create output tensor initialized to zero
    var out_tensor = try Tensor(outputType).fromShape(t1.allocator, t2.shape);

    try lean_sum_tensors(inputType, outputType, t1, t2, &out_tensor);

    return out_tensor;
}
// --------- lean SUM
pub inline fn lean_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *const Tensor(inputType), t2: *const Tensor(inputType), outputTensor: *Tensor(outputType)) !void {

    // Simple case: same size tensors
    if (t1.size == t2.size) {
        // Use unrolled loop for small sizes to avoid SIMD overhead
        if (t1.size <= 8) {
            comptime var unroll = 0;
            inline while (unroll < 8) : (unroll += 1) {
                if (unroll < t1.size and unroll < t2.size) {
                    outputTensor.data[unroll] = @as(outputType, t1.data[unroll] + t2.data[unroll]);
                }
            }
            return;
        }

        // Use SIMD for larger sizes
        const vector_len = std.simd.suggestVectorLength(inputType) orelse 4;
        const Vec = @Vector(vector_len, inputType);

        // Process 4 vectors at once to exploit instruction-level parallelism
        const chunk_size = vector_len * 4;
        const chunks = t1.size / chunk_size;
        var i: usize = 0;

        while (i < chunks * chunk_size) : (i += chunk_size) {
            inline for (0..4) |offset| {
                const v1: Vec = t1.data[i + offset * vector_len ..][0..vector_len].*;
                const v2: Vec = t2.data[i + offset * vector_len ..][0..vector_len].*;
                const result = v1 + v2;
                // Use a standard for loop instead of inline while for runtime execution
                for (0..vector_len) |j| {
                    outputTensor.data[i + offset * vector_len + j] = @as(outputType, result[j]);
                }
            }
        }

        // Handle remaining elements with simple loop
        while (i < t1.size) : (i += 1) {
            outputTensor.data[i] = @as(outputType, t1.data[i] + t2.data[i]);
        }
        return;
    }

    // Broadcasting case - use stack arrays for small ranks to avoid allocations
    const rank1 = t1.shape.len;
    const rank2 = t2.shape.len;
    const max_rank = @max(rank1, rank2);

    // Use stack arrays for common tensor ranks (up to 4D)
    var stack_shape1: [4]usize = undefined; // Initialize later
    var stack_shape2: [4]usize = undefined;
    var stack_strides1: [4]usize = undefined;
    var stack_strides2: [4]usize = undefined;
    var stack_out_strides: [4]usize = undefined;
    var stack_indices: [4]usize = [_]usize{0} ** 4;

    const shape1 = if (max_rank <= 4) stack_shape1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const shape2 = if (max_rank <= 4) stack_shape2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides1 = if (max_rank <= 4) stack_strides1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const strides2 = if (max_rank <= 4) stack_strides2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const out_strides = if (max_rank <= 4) stack_out_strides[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
    const indices = if (max_rank <= 4) stack_indices[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);

    // Only defer if we actually allocated
    if (max_rank > 4) {
        defer pkg_allocator.free(shape1);
        defer pkg_allocator.free(shape2);
        defer pkg_allocator.free(strides1);
        defer pkg_allocator.free(strides2);
        defer pkg_allocator.free(out_strides);
        defer pkg_allocator.free(indices);
    }

    // Special check: If t2 is 1D, try to find a matching dimension in t1
    var t2_target_dim: ?usize = null;
    if (rank2 == 1 and rank1 > 1) {
        const bias_len = t2.shape[0];
        for (0..rank1) |d| {
            if (t1.shape[d] == bias_len) {
                // Found a matching dimension. Assume this is the target for bias addition.
                // Align t2's dimension to this t1 dimension.
                // The dimension index needs to be relative to max_rank padding.
                t2_target_dim = max_rank - rank1 + d;
                //std.debug.print("\nINFO: Detected 1D t2 (size {}) matching t1 dim {} (at max_rank index {})\n", .{ bias_len, d, t2_target_dim.? });
                break;
            }
        }
    }

    // Initialize padded shapes with 1s before copying
    @memset(shape1, 1);
    @memset(shape2, 1);

    // Copy original shape1 from right to left
    var i: usize = 0;
    while (i < rank1) : (i += 1) {
        shape1[max_rank - rank1 + i] = t1.shape[i];
    }

    // Setup shape2: Special case for bias-like addition, otherwise copy from right-to-left
    if (t2_target_dim) |target_d| {
        // Special bias case: shape2 is all 1s except at target_d
        shape2[target_d] = t2.shape[0];
    } else {
        // Original logic: copy t2 shape from right-to-left
        i = 0; // Reset i
        while (i < rank2) : (i += 1) {
            shape2[max_rank - rank2 + i] = t2.shape[i];
        }
    }

    // Verify shapes and calculate output shape (DEBUG print included)
    //std.debug.print("\nBroadcasting Sum - Prepared shape1: {any}, shape2: {any}\n", .{ shape1, shape2 }); // DEBUG PRINT
    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            std.debug.print("Incompatible broadcast shapes at dim {}: {} vs {}\n", .{ dim, shape1[dim], shape2[dim] }); // DEBUG PRINT
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        outputTensor.shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Calculate strides from right to left
    var stride: usize = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        strides1[i] = if (shape1[i] > 1) stride else 0;
        strides2[i] = if (shape2[i] > 1) stride else 0;
        stride *= outputTensor.shape[i];
    }

    // Perform addition with broadcasting
    // Use stack arrays for common tensor ranks (up to 4D) - indices were already allocated above
    var stack_loop_indices: [4]usize = [_]usize{0} ** 4;
    const loop_indices = if (max_rank <= 4) stack_loop_indices[0..max_rank] else indices; // Reuse allocated 'indices' if max_rank > 4

    // Initialize loop_indices if using the stack allocation
    if (max_rank <= 4) {
        @memset(loop_indices, 0);
    }

    i = 0; // Reset i before the loop, don't redeclare
    while (i < outputTensor.size) : (i += 1) {
        // Calculate multi-dimensional indices for current output position 'i'
        var temp = i;
        for (0..max_rank) |dim| {
            const idx = max_rank - 1 - dim; // Iterate dimensions from right-to-left
            loop_indices[idx] = temp / out_strides[idx];
            temp = temp % out_strides[idx];
        }

        // Calculate linear input indices (idx1, idx2) using multi-dimensional indices and strides
        var idx1: usize = 0;
        var idx2: usize = 0;
        for (0..max_rank) |dim| {
            // stridesN[dim] is 0 if shapeN[dim] is 1, handling broadcasting implicitly
            idx1 += loop_indices[dim] * strides1[dim];
            idx2 += loop_indices[dim] * strides2[dim];
        }

        // Perform the addition
        outputTensor.data[i] = t1.data[idx1] + t2.data[idx2];
    }
}

/// Returns a Tensor with the same shape as the input tensors, where each element is the sum of all tensors at that location
pub fn sum_tensor_list(comptime inputType: anytype, comptime outputType: anytype, tensors: []const *const Tensor(inputType)) !Tensor(outputType) {
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;
    if (tensors.len == 1) {
        var out_tensor = try Tensor(outputType).fromShape(tensors[0].allocator, tensors[0].shape);
        for (0..tensors[0].data.len) |i| {
            out_tensor.data[i] = tensors[0].data[i];
        }
        return out_tensor;
    }

    // Use first tensor as reference for size and shape checks
    const ref_tensor = tensors[0];

    // Check all tensors have same size
    for (tensors[1..]) |t| {
        if (t.size != ref_tensor.size) return TensorMathError.InputTensorDifferentSize;
    }

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    }

    var out_tensor = try Tensor(outputType).fromShape(ref_tensor.allocator, ref_tensor.shape);
    try lean_sum_tensor_list(inputType, outputType, tensors, &out_tensor);

    return out_tensor;
}

pub inline fn lean_sum_tensor_list(comptime inputType: anytype, comptime outputType: anytype, tensors: []const *const Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Initialize output with first tensor
    for (0..tensors[0].data.len) |i| {
        outputTensor.data[i] = tensors[0].data[i];
    }

    // Add remaining tensors
    for (tensors[1..]) |t| {
        for (0..t.data.len) |i| {
            outputTensor.data[i] += t.data[i];
        }
    }
}
