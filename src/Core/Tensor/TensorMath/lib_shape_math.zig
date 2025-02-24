//! These operations change the structure or organization of a tensor.
//!
//!    Reshape: Change the shape without changing the data.
//!    Expand/Squeeze: like Padding and dilatation.
//!    Transpose/Permute: Change the order of dimensions.
//!    Flatten: Convert a multi-dimensional tensor into 1D.
//!    Concatenation: Combine tensors along a specified dimension.
//!    Split: Divide a tensor into smaller tensors.
//!    Flip: used to flip the kernel in some convolution operations.

const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const TensorError = @import("errorHandler").TensorError;

/// Resize the input tensor using interpolation.
/// Supports 'nearest', 'linear', and 'cubic' interpolation modes.
pub fn resize(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8) !Tensor(T) {
    if (scales == null and sizes == null) {
        return TensorError.InvalidInput;
    }
    if (scales != null and sizes != null) {
        return TensorError.InvalidInput;
    }

    // Calculate output dimensions
    var output_shape = try t.allocator.alloc(usize, t.shape.len);
    errdefer t.allocator.free(output_shape);

    if (scales) |s| {
        if (s.len != t.shape.len) {
            return TensorError.InvalidInput;
        }
        for (0..t.shape.len) |i| {
            output_shape[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(t.shape[i])) * s[i]));
        }
    } else if (sizes) |sz| {
        if (sz.len != t.shape.len) {
            return TensorError.InvalidInput;
        }
        @memcpy(output_shape, sz);
    }

    // Calculate total size of output tensor
    var total_size: usize = 1;
    for (output_shape) |dim| {
        total_size *= dim;
    }

    // Allocate memory for output data
    const output_data = try t.allocator.alloc(T, total_size);
    errdefer t.allocator.free(output_data);

    // Perform interpolation based on mode
    if (std.mem.eql(u8, mode, "nearest")) {
        try nearest_interpolation(T, t, output_data, output_shape, coordinate_transformation_mode);
    } else if (std.mem.eql(u8, mode, "linear")) {
        try linear_interpolation(T, t, output_data, output_shape, coordinate_transformation_mode);
    } else if (std.mem.eql(u8, mode, "cubic")) {
        try cubic_interpolation(T, t, output_data, output_shape, coordinate_transformation_mode);
    } else {
        return TensorError.UnsupportedMode;
    }

    return Tensor(T){
        .data = output_data,
        .shape = output_shape,
        .size = total_size,
        .allocator = t.allocator,
    };
}

pub fn get_resize_output_shape(input_shape: []const usize, scales: ?[]const f32, sizes: ?[]const usize) ![]usize {
    if (scales == null and sizes == null) {
        return TensorError.InvalidInput;
    }
    if (scales != null and sizes != null) {
        return TensorError.InvalidInput;
    }

    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    if (scales) |s| {
        if (s.len != input_shape.len) {
            return TensorError.InvalidInput;
        }
        for (0..input_shape.len) |i| {
            output_shape[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(input_shape[i])) * s[i]));
        }
    } else if (sizes) |sz| {
        if (sz.len != input_shape.len) {
            return TensorError.InvalidInput;
        }
        @memcpy(output_shape, sz);
    }

    return output_shape;
}

fn nearest_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    const input_strides = try self.getStrides();
    defer self.allocator.free(input_strides);
    const output_strides = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_strides);

    // Calculate output strides
    var stride: usize = 1;
    var idx: usize = output_shape.len;
    while (idx > 0) {
        idx -= 1;
        output_strides[idx] = stride;
        stride *= output_shape[idx];
    }

    var output_indices = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_indices);
    @memset(output_indices, 0);

    var done = false;
    while (!done) {
        var output_idx: usize = 0;
        var input_idx: usize = 0;

        for (0..output_shape.len) |i| {
            const scale = @as(f32, @floatFromInt(output_shape[i])) / @as(f32, @floatFromInt(self.shape[i]));
            var input_pos: f32 = undefined;

            if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                input_pos = (@as(f32, @floatFromInt(output_indices[i])) + 0.5) / scale - 0.5;
            } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                input_pos = @as(f32, @floatFromInt(output_indices[i])) * @as(f32, @floatFromInt(self.shape[i] - 1)) / @as(f32, @floatFromInt(output_shape[i] - 1));
            } else { // asymmetric
                input_pos = @as(f32, @floatFromInt(output_indices[i])) / scale;
            }

            const input_idx_i = @as(i32, @intFromFloat(@round(input_pos)));
            const clamped_idx = @min(@max(input_idx_i, 0), @as(i32, @intCast(self.shape[i] - 1)));
            input_idx += @as(usize, @intCast(clamped_idx)) * input_strides[i];
            output_idx += output_indices[i] * output_strides[i];
        }

        output_data[output_idx] = self.data[input_idx];

        // Increment indices
        done = true;
        for (0..output_shape.len) |i| {
            output_indices[output_shape.len - 1 - i] += 1;
            if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                done = false;
                break;
            }
            output_indices[output_shape.len - 1 - i] = 0;
        }
    }
}

fn linear_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    // For now, implement only for 1D and 2D tensors
    if (self.shape.len > 2) return TensorError.UnsupportedDimension;

    const input_strides = try self.getStrides();
    defer self.allocator.free(input_strides);

    var output_indices = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_indices);
    @memset(output_indices, 0);

    var done = false;
    while (!done) {
        var output_idx: usize = 0;
        if (output_shape.len == 1) {
            output_idx = output_indices[0];
        } else {
            output_idx = output_indices[0] * output_shape[1] + output_indices[1];
        }

        // Calculate interpolation coordinates
        var x: f32 = undefined;
        if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
            x = (@as(f32, @floatFromInt(output_indices[0])) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
        } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
            x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
        } else { // asymmetric
            x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
        }

        const x_floor = @floor(x);
        const x0 = @as(usize, @intFromFloat(@max(0, x_floor)));
        const x1 = @min(x0 + 1, self.shape[0] - 1);
        const dx = x - x_floor;

        if (self.shape.len == 1) {
            const v0 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0]))));
            const v1 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1]))));
            const interpolated = v0 * (1 - dx) + v1 * dx;
            output_data[output_idx] = @as(T, @intFromFloat(@round(interpolated)));
        } else {
            var y: f32 = undefined;
            if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                y = (@as(f32, @floatFromInt(output_indices[1])) + 0.5) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1])) - 0.5;
            } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1] - 1)) / @as(f32, @floatFromInt(output_shape[1] - 1));
            } else { // asymmetric
                y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1]));
            }

            const y_floor = @floor(y);
            const y0 = @as(usize, @intFromFloat(@max(0, y_floor)));
            const y1 = @min(y0 + 1, self.shape[1] - 1);
            const dy = y - y_floor;

            const v00 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0 * self.shape[1] + y0]))));
            const v01 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0 * self.shape[1] + y1]))));
            const v10 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1 * self.shape[1] + y0]))));
            const v11 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1 * self.shape[1] + y1]))));

            const tmp1 = v00 * (1 - dx) * (1 - dy);
            const tmp2 = v01 * (1 - dx) * dy;
            const tmp3 = v10 * dx * (1 - dy);
            const tmp4 = v11 * dx * dy;

            const interpolated = tmp1 + tmp2 + tmp3 + tmp4;
            output_data[output_idx] = @as(T, @intFromFloat(@round(interpolated)));
        }

        // Increment indices
        done = true;
        for (0..output_shape.len) |i| {
            output_indices[output_shape.len - 1 - i] += 1;
            if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                done = false;
                break;
            }
            output_indices[output_shape.len - 1 - i] = 0;
        }
    }
}

fn cubic_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    // For simplicity, implement only for 1D tensors initially
    if (self.shape.len != 1) return TensorError.UnsupportedDimension;

    var output_idx: usize = 0;
    while (output_idx < output_shape[0]) : (output_idx += 1) {
        var x: f32 = undefined;
        if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
            x = (@as(f32, @floatFromInt(output_idx)) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
        } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
            x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
        } else { // asymmetric
            x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
        }

        const x0 = @as(i32, @intFromFloat(@floor(x)));
        const dx = x - @as(f32, @floatFromInt(x0));

        var sum: f32 = 0;
        var weight_sum: f32 = 0;

        var i: i32 = -1;
        while (i < 3) : (i += 1) {
            const idx = x0 + i;
            if (idx >= 0 and idx < @as(i32, @intCast(self.shape[0]))) {
                const w = cubic_weight(dx - @as(f32, @floatFromInt(i)));
                sum += @as(f32, @floatFromInt(@as(i32, @intCast(self.data[@as(usize, @intCast(idx))])))) * w;
                weight_sum += w;
            }
        }

        output_data[output_idx] = @as(T, @intFromFloat(@round(sum / weight_sum)));
    }
}

fn cubic_weight(x: f32) f32 {
    const a = -0.75;
    const abs_x = @abs(x);
    if (abs_x <= 1) {
        return ((a + 2) * abs_x - (a + 3)) * abs_x * abs_x + 1;
    } else if (abs_x < 2) {
        return ((a * abs_x - 5 * a) * abs_x + 8 * a) * abs_x - 4 * a;
    }
    return 0;
}

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
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

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

pub fn get_concatenate_output_shape(tensors: []const []const usize, axis: isize) ![]usize {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].len;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have the same rank and matching shapes except along the concatenation axis
    for (tensors) |tensor_shape| {
        if (tensor_shape.len != rank) {
            return TensorError.MismatchedRank;
        }
        for (0..rank) |d| {
            if (d != concat_axis_usize and tensor_shape[d] != tensors[0][d]) {
                return TensorError.MismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    var new_shape = try pkg_allocator.alloc(usize, rank);
    errdefer pkg_allocator.free(new_shape);

    for (0..rank) |d| {
        if (d == concat_axis_usize) {
            var sum: usize = 0;
            for (tensors) |tensor_shape| {
                sum += tensor_shape[d];
            }
            new_shape[d] = sum;
        } else {
            new_shape[d] = tensors[0][d];
        }
    }

    return new_shape;
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

/// Given a 4D tensor it returns the tensor with the last 2 dimensions transposed. Operates on both data and shape, does not modify self, used by gemm.
pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {
    std.debug.print("\n[DEBUG] transposeLastTwo:", .{});
    std.debug.print("\n  Input tensor shape: ", .{});
    for (tensor.shape) |s| std.debug.print("{d} ", .{s});

    // Verifying correct shape
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        std.debug.print("\n  Error: Expected 2D or 4D tensor, got {d}D", .{tensor.shape.len});
        return TensorMathError.InputTensorsWrongShape;
    }

    var rows: usize = undefined;
    var cols: usize = undefined;
    var total: usize = undefined;
    var newShape: []usize = undefined;

    if (tensor.shape.len == 2) {
        rows = tensor.shape[0];
        cols = tensor.shape[1];
        total = rows * cols;
        newShape = try pkg_allocator.alloc(usize, 2);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = cols;
        newShape[1] = rows;
    } else { // 4D case
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        rows = tensor.shape[2];
        cols = tensor.shape[3];
        total = batch * channel * rows * cols;
        newShape = try pkg_allocator.alloc(usize, 4);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = batch;
        newShape[1] = channel;
        newShape[2] = cols;
        newShape[3] = rows;
    }

    std.debug.print("\n  Rows: {d}, Cols: {d}, Total: {d}", .{ rows, cols, total });
    std.debug.print("\n  New shape: ", .{});
    for (newShape) |s| std.debug.print("{d} ", .{s});

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

    std.debug.print("\n  Transposing data...", .{});

    if (tensor.shape.len == 2) {
        // Simple 2D transpose - Fixed indexing
        for (0..rows) |i| {
            for (0..cols) |j| {
                outData[j * rows + i] = tensor.data[i * cols + j];
            }
        }
    } else {
        // 4D transpose of last two dimensions
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        for (0..batch) |b| {
            for (0..channel) |c| {
                for (0..rows) |i| {
                    for (0..cols) |j| {
                        const index_in = (((b * channel) + c) * rows + i) * cols + j;
                        const index_out = (((b * channel) + c) * cols + j) * rows + i;
                        outData[index_out] = tensor.data[index_in];
                    }
                }
            }
        }
    }

    std.debug.print("\n  Transpose complete", .{});

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
    };
}

/// Method to add a top&bottom padding and a left&right padding.
/// At the moment the function only supports 2 padding params, but the method
/// is already set to have different left, right, top and bottom padding values.
pub fn addPaddingAndDilation(
    comptime T: type,
    t: *Tensor(T),
    upDownPadding: usize,
    leftRightPadding: usize,
    verticalDil: usize,
    horizontalDil: usize,
) !void {

    //checks on padding dim (usize is alway >= 0)
    if (t.shape.len < 2) return TensorError.TooSmallToPadding;

    const upPadding = upDownPadding;
    const downPadding = upDownPadding;
    const leftPadding = leftRightPadding;
    const rightPadding = leftRightPadding;
    const dim = t.shape.len;

    const new_row_numb = t.shape[dim - 2] + upPadding + downPadding + verticalDil * (t.shape[dim - 2] - 1);
    const new_col_numb = t.shape[dim - 1] + leftPadding + rightPadding + horizontalDil * (t.shape[dim - 1] - 1);
    //std.debug.print("\n new_row_numb: {} new_col_numb:{}", .{ new_row_numb, new_col_numb });

    //compute new shape
    const new_shape = try t.allocator.alloc(usize, dim);
    @memcpy(new_shape, t.shape);
    new_shape[dim - 1] = new_col_numb;
    new_shape[dim - 2] = new_row_numb;

    //compute new size
    var new_total_size: usize = 1;
    for (new_shape) |size_i| {
        new_total_size *= size_i;
    }

    //alloc new tensor.data memory space to all zero
    const new_data = try t.allocator.alloc(T, new_total_size);
    @memset(new_data, 0);

    const new_matrix_dim = new_row_numb * new_col_numb;
    const total_number_2DMatrices = new_total_size / new_matrix_dim;
    const old_matrix_dim = t.shape[dim - 2] * t.shape[dim - 1];
    const old_total_number_2DMatrices = t.size / old_matrix_dim; //just for check assertion
    std.debug.assert(total_number_2DMatrices == old_total_number_2DMatrices);

    for (0..total_number_2DMatrices) |matix_i| {
        const num_elem_prec_new_matr = matix_i * new_matrix_dim;
        const num_elem_prec_old_matr = matix_i * old_matrix_dim;
        var i = upPadding;
        var old_row: usize = 0;
        while (i < new_row_numb - downPadding) : (i += (1 + verticalDil)) {
            var j = leftPadding;
            var old_col: usize = 0;
            while (j < new_col_numb - rightPadding) : (j += (1 + horizontalDil)) {
                const idx_new_matr = num_elem_prec_new_matr + i * new_col_numb + j;
                const idx_old_matr = num_elem_prec_old_matr + old_row * (t.shape[dim - 1]) + old_col;
                new_data[idx_new_matr] = t.data[idx_old_matr];
                old_col += 1;
            }
            old_row += 1;
        }
    }

    //free all old attributes and setting new ones
    t.allocator.free(t.data);
    t.allocator.free(t.shape);

    t.shape = new_shape;
    t.data = new_data;
    t.size = new_total_size;
}

/// Helper function to flip (rotate 180 degrees horizontaly and vertically) the kernel in convolution or any other matix 2D
/// ex:
///  flip( [[a, b], [c, d], [e, f]] ) = [[f, e], [d, c], [b, a]]
pub fn flip(comptime T: type, kernel: *Tensor(T)) !Tensor(T) {
    const kernel_dim = kernel.shape.len;
    const kernel_row = kernel.shape[kernel_dim - 2];
    const kernel_cols = kernel.shape[kernel_dim - 1];
    const matrix_dim = kernel_cols * kernel_row;

    //create and initialize the new shape
    const flipped_shape = try kernel.allocator.alloc(usize, kernel.shape.len);
    defer kernel.allocator.free(flipped_shape);
    @memcpy(flipped_shape, kernel.shape);

    var flipped_kernel = try Tensor(T).fromShape(kernel.allocator, flipped_shape);

    const total_number_2DMatrices = flipped_kernel.size / matrix_dim;

    for (0..total_number_2DMatrices) |matix_i| {
        for (0..kernel_row) |i| {
            for (0..kernel_cols) |j| {
                flipped_kernel.data[(matix_i + 1) * matrix_dim - (i * kernel_cols + j + 1)] = kernel.data[matix_i * matrix_dim + i * kernel_cols + j];
            }
        }
    }

    return flipped_kernel;
}

/// Split a tensor into multiple tensors along a specified axis.
/// If split_sizes is null, the tensor is split into equal parts.
/// If split_sizes is provided, it specifies the size of each split.
/// Negative axis values count from the back (-1 means last axis).
/// Returns an array of tensors that must be freed by the caller.
pub fn split(comptime T: anytype, t: *Tensor(T), axis: i64, split_sizes: ?[]const usize) ![]Tensor(T) {
    // Handle negative axis
    const positive_axis = @as(usize, @intCast(if (axis < 0) @as(i64, @intCast(t.shape.len)) + axis else axis));
    if (positive_axis >= t.shape.len) return TensorError.InvalidAxis;

    // Calculate split sizes
    const dim_size = t.shape[positive_axis];
    var sizes = std.ArrayList(usize).init(t.allocator.*);
    defer sizes.deinit();

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: usize = 0;
        for (s) |size| {
            try sizes.append(size);
            total_size += size;
        }
        if (total_size != dim_size) return TensorError.InvalidSplitSize;
    } else {
        // Split into equal parts
        if (dim_size == 0) return TensorError.InvalidSplitSize;
        const split_size = dim_size;
        try sizes.append(split_size);
    }

    // Create output tensors
    var output_tensors = try t.allocator.alloc(Tensor(T), sizes.items.len);
    errdefer {
        for (output_tensors) |*tensor| {
            tensor.deinit();
        }
        t.allocator.free(output_tensors);
    }

    var offset: usize = 0;
    for (sizes.items, 0..) |split_size, i| {
        // Create shape for the split tensor
        var new_shape = try t.allocator.alloc(usize, t.shape.len);
        errdefer t.allocator.free(new_shape);
        @memcpy(new_shape, t.shape);
        new_shape[positive_axis] = split_size;

        // Calculate total size for the split tensor
        var total_size: usize = 1;
        for (new_shape) |dim| {
            total_size *= dim;
        }

        // Allocate memory for the split tensor's data
        var new_data = try t.allocator.alloc(T, total_size);
        errdefer t.allocator.free(new_data);

        // Calculate strides
        var stride: usize = 1;
        for (positive_axis + 1..t.shape.len) |j| {
            stride *= t.shape[j];
        }

        // Copy data to the split tensor
        const block_size = split_size * stride;
        const num_blocks = total_size / block_size;

        var block_idx: usize = 0;
        while (block_idx < num_blocks) : (block_idx += 1) {
            const src_start = offset + block_idx * dim_size * stride;
            const dst_start = block_idx * split_size * stride;
            const copy_size = split_size * stride;
            @memcpy(new_data[dst_start .. dst_start + copy_size], t.data[src_start .. src_start + copy_size]);
        }

        // Create the split tensor
        output_tensors[i] = .{
            .data = new_data,
            .size = total_size,
            .shape = new_shape,
            .allocator = t.allocator,
        };

        offset += split_size * stride;
    }

    return output_tensors;
}

pub fn get_split_output_shapes(input_shape: []const usize, axis: i64, split_sizes: ?[]const usize) ![][]usize {
    // Handle negative axis
    var positive_axis: usize = undefined;
    if (axis < 0) {
        const adjusted = @as(i64, @intCast(input_shape.len)) + axis;
        if (adjusted < 0) return TensorError.InvalidAxis;
        positive_axis = @intCast(adjusted);
    } else {
        positive_axis = @intCast(axis);
    }

    if (positive_axis >= input_shape.len) return TensorError.InvalidAxis;

    // Rest of the function remains the same...
    const dim_size = input_shape[positive_axis];
    var sizes = std.ArrayList(usize).init(pkg_allocator);
    defer sizes.deinit();

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: usize = 0;
        for (s) |size| {
            try sizes.append(size);
            total_size += size;
        }
        if (total_size != dim_size) return TensorError.InvalidSplitSize;
    } else {
        // Split into equal parts
        if (dim_size == 0) return TensorError.InvalidSplitSize;
        const split_size = dim_size;
        try sizes.append(split_size);
    }

    // Create output shapes
    var output_shapes = try pkg_allocator.alloc([]usize, sizes.items.len);
    errdefer {
        for (output_shapes) |shape| {
            pkg_allocator.free(shape);
        }
        pkg_allocator.free(output_shapes);
    }

    // Fill output shapes
    for (sizes.items, 0..) |split_size, i| {
        output_shapes[i] = try pkg_allocator.alloc(usize, input_shape.len);
        errdefer pkg_allocator.free(output_shapes[i]);

        @memcpy(output_shapes[i], input_shape);
        output_shapes[i][positive_axis] = split_size;
    }

    return output_shapes;
}

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// The lean version of this method follows the onnx standard.
/// https://onnx.ai/onnx/operators/onnx__Reshape.html
pub fn reshape(comptime T: anytype, input: *Tensor(T), newShape: []usize, allowZero: ?bool) !Tensor(T) {
    //TODO: threat allowZero properly
    var total_size: usize = 1;
    for (newShape) |dim| {
        total_size *= dim;
    }
    if (total_size != input.size) {
        return TensorError.InputArrayWrongSize;
    }

    var output = try Tensor(T).fromShape(&pkg_allocator, newShape);

    try reshape_lean(T, input, newShape, allowZero, &output);

    return output;
}

/// lean version of the above reshape
pub fn reshape_lean(comptime T: anytype, input: *Tensor(T), newShape: []usize, allowZero: ?bool, output: *Tensor(T)) !void {
    _ = allowZero; //TODO: threat allowZero properly

    // std.debug.print("\nReshape Debug:", .{});
    // std.debug.print("\n  Input shape: ", .{});
    // for (input.shape) |s| {
    //     std.debug.print("{d} ", .{s});
    // }
    // std.debug.print("\n  New shape: ", .{});
    // for (newShape) |s| {
    //     std.debug.print("{d} ", .{s});
    // }

    // Calculate total size of new shape
    var total_size: usize = 1;
    for (newShape) |dim| {
        total_size *= dim;
    }
    // std.debug.print("\n  Total size from new shape: {d}", .{total_size});
    // std.debug.print("\n  Input size: {d}", .{input.size});
    // std.debug.print("\n  Output data len: {d}", .{output.data.len});
    // std.debug.print("\n  Input data len: {d}", .{input.data.len});

    // Verify sizes match
    if (total_size != input.size) {
        //std.debug.print("\n  Error: Size mismatch!", .{});
        return TensorError.InputArrayWrongSize;
    }

    // Update output shape
    for (newShape, 0..) |dim, i| {
        output.shape[i] = dim;
    }
    output.size = total_size;

    // Copy data only if sizes match
    if (output.data.len == input.data.len) {
        @memcpy(output.data, input.data);
        //std.debug.print("\n  Data copied successfully", .{});
    } else {
        //std.debug.print("\n  Error: Data length mismatch!", .{});
        return TensorError.InputArrayWrongSize;
    }
}

/// Implements https://onnx.ai/onnx/operators/onnx__Gather.html
/// NOTE: (IMPORTANT FOR CODE GEN) according to onnx standard, values in indices tensor can be negative and if so they are converted to positive values by adding the size of the axis pointed dimension of the data tensor. For performance and code clarity reasons (check + double casting) we support only positive indices instead, remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
/// Gather elements from the data tensor along the specified axis using the provided indices.
/// The axis parameter specifies the axis along which the elements will be gathered.
/// The shape of the output tensor is the same as the shape of the data tensor, with the axis dimension replaced with the shape of the indices tensor.
/// The output tensor is created by copying elements from the input tensor using the indices tensor.
pub fn gather(comptime T: anytype, data: *Tensor(T), indices: *Tensor(usize), selected_axis: isize) !Tensor(T) {
    // Scalar data tensor is not allowed
    if (data.shape.len == 0) {
        return TensorError.InvalidRank;
    }

    // Validate that the axis is within the tensor's dimensions
    const number_dimensions: isize = @intCast(data.shape.len);
    if (selected_axis >= number_dimensions or selected_axis < -1 * number_dimensions) {
        return TensorError.InvalidAxis;
    }

    // If axis is negative, convert it to a positive index
    const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

    // All index values must be within bounds [0, s-1] where s is the length of the chosen axis
    for (0..indices.size) |i| {
        if (indices.data[i] >= data.shape[axis] or indices.data[i] < 0) {
            return TensorError.IndexOutOfBounds;
        }
    }

    // Calculate the shape of the output tensor:
    // [data.shape[0..axis], indices.shape..., data.shape[axis+1..]]
    const output_shape_len = data.shape.len + indices.shape.len - 1;
    const output_shape = try pkg_allocator.alloc(usize, output_shape_len);
    defer pkg_allocator.free(output_shape);
    errdefer pkg_allocator.free(output_shape);

    // Copy the dimensions before the axis
    for (0..axis) |i| {
        output_shape[i] = data.shape[i];
    }

    // Copy indices shape
    var indices_idx: usize = 0;
    while (indices_idx < indices.shape.len) : (indices_idx += 1) {
        output_shape[axis + indices_idx] = indices.shape[indices_idx];
    }

    // Copy the dimensions after the axis
    for (axis + 1..data.shape.len) |i| {
        output_shape[axis + indices.shape.len + (i - axis - 1)] = data.shape[i];
    }

    // Calculate total size
    var total_size: usize = 1;
    for (output_shape) |dim| {
        total_size *= dim;
    }

    // Create output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try lean_gather(T, data, indices, selected_axis, &output);

    return output;
}

/// Lean version of gather
/// NOTE: (IMPORTANT FOR CODE GEN) according to onnx standard, values in indices tensor can be negative and if so they are converted to positive values by adding the size of the axis pointed dimension of the data tensor. For performance and code clarity reasons (check + double casting) we support only positive indices instead, remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
pub fn lean_gather(comptime T: anytype, data: *Tensor(T), indices: *Tensor(usize), selected_axis: isize, output: *Tensor(T)) !void {
    std.debug.print("\n[GATHER DEBUG] Data shape: ", .{});
    for (data.shape) |s| std.debug.print("{d} ", .{s});

    std.debug.print("\n[GATHER DEBUG] Indices shape: ", .{});
    for (indices.shape) |s| std.debug.print("{d} ", .{s});

    std.debug.print("\n[GATHER DEBUG] Output shape: ", .{});
    for (output.shape) |s| std.debug.print("{d} ", .{s});

    std.debug.print("\n[GATHER DEBUG] Selected axis: {d}\n", .{selected_axis});

    //If axis is negative, convert it to a positive index
    const number_dimensions: isize = @intCast(data.shape.len);
    const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

    // Compute the total number of elements in each segment
    var outer_size: usize = 1;
    for (0..axis) |i| outer_size *= data.shape[i];

    const indices_size: usize = indices.size;

    var inner_size: usize = 1;
    for ((axis + 1)..data.shape.len) |i| {
        inner_size *= data.shape[i];
    }

    std.debug.print("[GATHER DEBUG] Outer size: {d}, Inner size: {d}, Indices size: {d}\n", .{ outer_size, inner_size, indices_size });

    // Iterate over each "outer" segment
    for (0..outer_size) |outer_idx| {
        // Iterate over each index in the indices tensor
        for (0..indices_size) |idx| {
            // Retrieve the gather index from the indices tensor
            const gather_idx = try indices.get(idx);

            // Calculate the correct data_offset
            const data_offset = (outer_idx * data.shape[axis] + gather_idx) * inner_size;

            // Calculate the starting offset in the output tensor
            const output_offset = (outer_idx * indices_size + idx) * inner_size;

            std.debug.print("[GATHER DEBUG] Outer idx: {d}, Gather idx: {d}, Data offset: {d}, Output offset: {d}\n", .{ outer_idx, gather_idx, data_offset, output_offset });

            // Perform the data copy using std.mem.copy
            @memcpy(output.data[output_offset .. output_offset + inner_size], data.data[data_offset .. data_offset + inner_size]);

            std.debug.print("[GATHER DEBUG] Copied data: ", .{});
            for (data.data[data_offset .. data_offset + inner_size]) |val| {
                std.debug.print("{d} ", .{val});
            }
            std.debug.print("\n", .{});
        }
    }
}
