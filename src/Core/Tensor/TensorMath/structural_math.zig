const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const TensorError = @import("errorHandler").TensorError;

/// Concatenates a list of tensors into a single tensor along the specified axis.
/// All input tensors must have the same shape, except for the size of the concatenation axis.
///
/// Parameters:
///     allocator - The memory allocator to use for the new tensor.
///     tensors - An array of tensors to concatenate.
///     axis - The axis along which to concatenate. Negative values count dimensions from the back.
///
/// Returns:
///     A new tensor resulting from concatenation.
///
/// Errors:
///     - TensorError.EmptyTensorList
///     - TensorError.AxisOutOfBounds
///     - TensorError.MismatchedRank
///     - TensorError.MismatchedShape
pub fn concatenate(comptime T: type, allocator: *std.mem.Allocator, tensors: []Tensor(T), axis: isize) !Tensor(T) {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorError.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].shape.len;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have the same rank and matching shapes except along the concatenation axis
    for (tensors) |tensor| {
        if (tensor.shape.len != rank) {
            return TensorError.MismatchedRank;
        }
        for (0..rank) |d| {
            if (d != concat_axis_usize and tensor.shape[d] != tensors[0].shape[d]) {
                return TensorError.MismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    var new_shape = try allocator.alloc(usize, rank);
    for (0..rank) |d| {
        if (d == concat_axis_usize) {
            var sum: usize = 0;
            for (tensors) |tensor| {
                sum += tensor.shape[d];
            }
            new_shape[d] = sum;
        } else {
            new_shape[d] = tensors[0].shape[d];
        }
    }

    // Calculate the total number of elements in the new tensor
    var total_size: usize = 1;
    for (new_shape) |dim| {
        total_size *= dim;
    }

    // Allocate memory for the new tensor's data
    var new_data = try allocator.alloc(T, total_size);

    // Calculate the number of slices based on the concatenation axis
    var num_slices: usize = 1;
    for (0..concat_axis_usize) |d| {
        num_slices *= new_shape[d];
    }

    // Calculate the slice size (number of elements to copy per concatenation dimension)
    var slice_size: usize = 1;
    if (concat_axis_usize + 1 < rank) {
        for ((concat_axis_usize + 1)..rank) |d| {
            slice_size *= new_shape[d];
        }
    } else {
        slice_size = 1;
    }

    // Initialize the offset for copying data into new_data
    var offset: usize = 0;

    // Iterate over each slice
    for (0..num_slices) |slice_idx| {
        for (tensors, 0..) |tensor, tensor_idx| {
            const concat_dim = tensor.shape[concat_axis_usize];
            const copy_size = concat_dim * slice_size;

            std.debug.print("\n  Copying Tensor {}: slice_idx={} concat_dim={} slice_size={} copy_size={} to new_data[{}..{}]", .{ tensor_idx, slice_idx, concat_dim, slice_size, copy_size, offset, offset + copy_size });

            // Calculate the start and end indices in the source tensor
            const src_start = slice_idx * concat_dim * slice_size;
            const src_end = src_start + copy_size;

            // Check bounds for the source tensor's data
            if (src_end > tensor.data.len) {
                std.debug.print("\n  Out of bounds error for tensor idx:{} src_end:{} tensor.data.len:{}", .{ tensor_idx, src_end, tensor.data.len });
                return TensorError.IndexOutOfBounds;
            }

            // Calculate the destination indices in new_data
            const dest_start = offset;
            const dest_end = offset + copy_size;

            // Check bounds for the new_data buffer
            if (dest_end > new_data.len) {
                std.debug.print("\n  Out of bounds error for new_data dest_end:{} new_data.len:{}", .{ dest_end, new_data.len });
                return TensorError.IndexOutOfBounds;
            }

            @memcpy(new_data[dest_start..dest_end], tensor.data[src_start..src_end]);

            // Update the offset for the next copy
            offset += copy_size;
        }
    }

    // Return the concatenated tensor
    return Tensor(T){
        .data = new_data,
        .size = total_size,
        .shape = new_shape,
        .allocator = allocator,
    };
}

/// Calculate strides for a given shape
pub fn calculateStrides(shape: []usize, allocator: *const std.mem.Allocator) ![]usize {
    const len = shape.len;
    const strides = try allocator.alloc(usize, len);
    if (len == 0) return strides; // Handle scalar tensor
    strides[len - 1] = 1;
    for (1..len) |i| {
        strides[len - 1 - i] = strides[len - i] * shape[len - i];
    }
    return strides;
}

/// Returns a Tensor self transposed. Does not modify self.
/// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
pub fn transpose2D(comptime T: type, t: *Tensor(T)) !Tensor(T) {
    if (t.shape.len != 2) {
        return error.InvalidDimension; // For simplicity, let's focus on 2D for now
    }

    const allocator = t.allocator;

    // Shape of the transposed tensor
    const transposed_shape: [2]usize = [_]usize{ t.shape[1], t.shape[0] };
    const tensorShape = try allocator.alloc(usize, t.shape.len);
    @memcpy(tensorShape, &transposed_shape);

    // Allocate space for transposed data
    const transposed_data = try allocator.alloc(T, t.size);

    // Perform the transposition
    for (0..t.shape[0]) |i| {
        for (0..t.shape[1]) |j| {
            // For 2D tensor, flatten the index and swap row/column positions
            const old_idx = i * t.shape[1] + j;
            const new_idx = j * t.shape[0] + i;
            transposed_data[new_idx] = t.data[old_idx];
        }
    }

    return Tensor(T){
        .data = transposed_data,
        .size = t.size,
        .shape = tensorShape,
        .allocator = allocator,
    };
}

/// Returns a Tensor self transposed.
/// OSS! Does not modify self.data!! it only changes the shape! so it is necessary to acces it trough get_flatten_index()
/// By default, it transposes the tensor to the reverse shape.
pub fn transposeDefault(comptime T: type, t: *Tensor(T)) !Tensor(T) {
    // Reverse the shape of the tensor
    const tensorShape = try t.allocator.alloc(usize, t.shape.len);
    for (0..t.shape.len) |i| {
        tensorShape[i] = t.shape.len - 1 - i;
    }

    return transpose(T, t, tensorShape);
}

/// Returns a Tensor self transposed. Does not modify self.
fn transpose(comptime T: type, t: *Tensor(T), perms: []usize) !Tensor(T) {
    defer t.allocator.free(perms);
    const num_dims = t.shape.len;
    if (perms.len != num_dims) {
        return error.InvalidDimension;
    }

    // Check that the permutation is valid
    var bitmap = try t.allocator.alloc(bool, perms.len);
    defer t.allocator.free(bitmap);

    for (perms) |perm| {
        if (perm >= perms.len) {
            return error.InvalidPermutation;
        }
        if (bitmap[perm] == true) {
            return error.InvalidPermutation;
        }
        bitmap[perm] = true;
    }

    // Allocate space for the new shape
    const new_shape = try t.allocator.alloc(usize, num_dims);
    for (0..num_dims) |i| {
        new_shape[i] = t.shape[perms[i]];
    }
    defer t.allocator.free(new_shape);

    // Create the new tensor
    const new_tensor = try Tensor(T).fromShape(t.allocator, new_shape);

    // Copy data to the new tensor
    for (0..t.size) |i| {
        new_tensor.data[i] = t.data[i];
    }

    return new_tensor;
}
