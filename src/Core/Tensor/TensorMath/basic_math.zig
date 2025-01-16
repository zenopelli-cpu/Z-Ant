const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const TensorError = @import("errorHandler").TensorError;
const Architectures = @import("architectures").Architectures; //Import Architectures type
const Converter = @import("typeC");
const ArchitectureError = @import("errorHandler").ArchitectureError;

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

    // Instead of using threads, just do it directly
    var index: usize = 0;
    while (index < tensor.size) : (index += len) {
        for (0..len) |i| {
            tensor.data[index + i] += bias.data[i];
        }
    }
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

/// Performs element-wise binary subtraction with Numpy-style broadcasting support
pub fn sub_tensors(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !Tensor(Tout) {
    //selecting between all possible architectures
    return switch (arch) {
        Architectures.CPU => return CPU_sub_tensors(Tin, Tout, t1, t2),
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

fn CPU_sub_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
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

    // Verify shapes are compatible for broadcasting
    var out_shape = try pkg_allocator.alloc(usize, max_rank);
    errdefer pkg_allocator.free(out_shape);

    for (0..max_rank) |dim| {
        if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }
        out_shape[dim] = @max(shape1[dim], shape2[dim]);
    }

    // Calculate total size and strides
    var total_size: usize = 1;
    var strides1 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(strides1);
    var strides2 = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(strides2);
    var out_strides = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(out_strides);

    // Calculate strides from right to left
    var stride: usize = 1;
    i = max_rank;
    while (i > 0) {
        i -= 1;
        out_strides[i] = stride;
        strides1[i] = if (shape1[i] > 1) stride else 0;
        strides2[i] = if (shape2[i] > 1) stride else 0;
        stride *= out_shape[i];
    }
    total_size = stride;

    std.debug.print("\nshape1: {any}, strides1: {any}", .{ shape1, strides1 });
    std.debug.print("\nshape2: {any}, strides2: {any}", .{ shape2, strides2 });
    std.debug.print("\nout_shape: {any}, out_strides: {any}", .{ out_shape, out_strides });

    // Allocate output tensor
    var out_data = try pkg_allocator.alloc(outputType, total_size);
    errdefer pkg_allocator.free(out_data);

    // Perform subtraction with broadcasting
    var indices = try pkg_allocator.alloc(usize, max_rank);
    defer pkg_allocator.free(indices);
    @memset(indices, 0);

    i = 0;
    while (i < total_size) : (i += 1) {
        // Calculate indices for current position
        var temp = i;
        for (0..max_rank) |dim| {
            const idx = max_rank - 1 - dim;
            indices[idx] = temp / out_strides[idx];
            temp = temp % out_strides[idx];
        }

        // Calculate input indices considering broadcasting
        var idx1: usize = 0;
        var idx2: usize = 0;

        // For same shape tensors, use the same index calculation
        if (std.mem.eql(usize, shape1, shape2)) {
            idx1 = i;
            idx2 = i;
        } else {
            // For broadcasting case
            for (0..max_rank) |dim| {
                const pos = indices[dim];
                // For t1: if dimension is 1, don't increment index (broadcasting)
                if (shape1[dim] > 1) {
                    idx1 += pos * strides1[dim];
                }
                // For t2: if dimension is 1, don't increment index (broadcasting)
                if (shape2[dim] > 1) {
                    const t2_pos = pos % shape2[dim];
                    idx2 += t2_pos * strides2[dim];
                }
            }
        }

        // Perform subtraction: t1 - t2
        const val1 = t1.data[idx1];
        const val2 = t2.data[idx2];
        out_data[i] = val1 - val2;
        std.debug.print("\nFinal: idx1={}, val1={}, idx2={}, val2={}, result={}", .{ idx1, val1, idx2, val2, out_data[i] });
    }

    return Tensor(outputType){
        .data = out_data,
        .shape = out_shape,
        .size = total_size,
        .allocator = &pkg_allocator,
    };
}

pub fn mul(comptime T: anytype, lhs: *Tensor(T), rhs: *Tensor(T)) !Tensor(T) {
    if (lhs.size != rhs.size) {
        return TensorError.MismatchedShape;
    }

    const allocator = lhs.allocator;
    const result = try Tensor(T).fromShape(allocator, lhs.shape);

    for (0..lhs.size) |i| {
        result.data[i] = lhs.data[i] * rhs.data[i];
    }

    return result;
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

pub fn equal(comptime T: anytype, t1: *Tensor(T), t2: *Tensor(T)) bool {
    //same size
    if (t1.size != t2.size) {
        std.debug.print("\n\n ERROR:WRONG SIZE t1.size:{} t2.size:{}", .{ t1.size, t2.size });
        return false;
    }

    //same shape
    for (0..t1.shape.len) |i| {
        if (t1.shape[i] != t2.shape[i]) {
            std.debug.print("\n\n ERROR: WRONG SHAPE t1.shape[{}]:{} t2.shape[{}]:{}", .{ i, t1.shape[i], i, t2.shape[i] });
            return false;
        }
    }

    //same data
    if (!std.mem.eql(T, t1.data, t2.data)) {
        std.debug.print("\n\n ERROR: WRONG DATA", .{});
        return false;
    }

    return true;
}

/// Returns true if the Tensor is one-hot encoded
fn isOneHot(comptime T: anytype, t: *Tensor(T)) !bool {
    const elems_row = t.shape[t.shape.len - 1];
    if (elems_row == 0) {
        return TensorError.EmptyTensor;
    }
    const numb_rows = t.size / elems_row;
    if (numb_rows == 0) {
        return TensorError.ZeroSizeTensor;
    }

    for (0..numb_rows) |row| {
        var oneHotFound = false;
        for (0..t.shape[t.shape.len - 1]) |i| {
            if (t.data[row * elems_row + i] == 1 and !oneHotFound) {
                if (!oneHotFound) oneHotFound = true else return TensorError.NotOneHotEncoded;
            }
        }
    }

    return true;
}

/// Returns true only if all the values of shape and data are valid numbers
pub fn isSafe(comptime T: anytype, t: *Tensor(T)) !void {
    switch (@typeInfo(T)) {
        .Float => {
            // Loop over tensor data
            for (t.data) |*value| {
                if (std.math.isNan(value.*)) return TensorError.NanValue;
                if (!std.math.isFinite(value.*)) return TensorError.NotFiniteValue;
            }

            // Loop over tensor shape
            for (t.shape) |*value| {
                if (std.math.isNan(value.*)) return TensorError.NanValue;
            }
        },
        else => {
            // If T is not Float, skip isSafe checks
            return;
        },
    }
}
