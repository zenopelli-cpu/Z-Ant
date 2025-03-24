//! These operations reduce a tensor to a smaller shape by aggregating values. Like:
//!    Sum: Compute the sum of elements along specific dimensions.
//!   Mean: Compute the average.
//!    Min/Max: Find the minimum or maximum value.
//!    Prod: Compute the product of elements.
//!    Standard Deviation and Variance: Statistical operations.
//!
const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const Converter = zant.utils.type_converter;

/// Performs the mean of a given tensor. It is a reduction operation, collapsing the whole tenosr into a single value.
pub fn mean(comptime T: anytype, tensor: *Tensor(T)) f32 {
    var res: f32 = 0;

    for (tensor.data) |*d| {
        res += Converter.convert(T, f32, d.*);
    }
    res = res / Converter.convert(usize, f32, tensor.size);
    return res;
}

/// Computes the mean of the input tensor's elements along the provided axes.
/// The resulting tensor has the same rank as the input if keepdims equals true.
/// If keepdims equals false, then the resulting tensor has the reduced dimension pruned.
pub fn reduce_mean(comptime T: anytype, tensor: *Tensor(T), axes: ?[]const i64, keepdims: bool, noop_with_empty_axes: bool) !Tensor(T) {
    // Calculate output shape first
    var out_shape: []usize = undefined;
    const allocator = tensor.allocator;

    if (axes == null or axes.?.len == 0) {
        if (noop_with_empty_axes) {
            return tensor.copy();
        }
        // Reduce over all dimensions
        if (keepdims) {
            out_shape = try allocator.alloc(usize, tensor.shape.len);
            @memset(out_shape, 1);
        } else {
            out_shape = try allocator.alloc(usize, 1);
            out_shape[0] = 1;
        }
    } else {
        // Mark dimensions to reduce
        var reduce_dims = try allocator.alloc(bool, tensor.shape.len);
        defer allocator.free(reduce_dims);
        @memset(reduce_dims, false);

        for (axes.?) |axis| {
            const actual_axis = if (axis < 0)
                @as(usize, @intCast(@as(i64, @intCast(tensor.shape.len)) + axis))
            else
                @as(usize, @intCast(axis));
            reduce_dims[actual_axis] = true;
        }

        // Count remaining dimensions and create output shape
        var remaining_dims: usize = 0;
        for (reduce_dims) |is_reduced| {
            if (!is_reduced or keepdims) remaining_dims += 1;
        }

        out_shape = try allocator.alloc(usize, remaining_dims);
        var out_dim: usize = 0;
        for (tensor.shape, 0..) |dim_size, i| {
            if (!reduce_dims[i] or keepdims) {
                out_shape[out_dim] = if (reduce_dims[i]) 1 else dim_size;
                out_dim += 1;
            }
        }
    }

    // Create output tensor
    var output = try Tensor(T).fromShape(allocator, out_shape);
    allocator.free(out_shape);
    errdefer output.deinit();

    try lean_reduce_mean(T, tensor, &output, axes, keepdims, noop_with_empty_axes);
    return output;
}

/// Lean version of reduce_mean that operates on pre-allocated output tensor
pub inline fn lean_reduce_mean(
    comptime T: anytype,
    input_tensor: *Tensor(T),
    output_tensor: *Tensor(T),
    axes: ?[]const i64,
    _: bool, // keepdims (unused since output shape is pre-determined)
    noop_with_empty_axes: bool,
) !void {
    // Handle empty axes case
    if (axes == null or axes.?.len == 0) {
        if (noop_with_empty_axes) {
            // Act as identity operation
            @memcpy(output_tensor.data, input_tensor.data);
            return;
        }
        // Reduce over all dimensions
        var sum: T = 0;
        for (input_tensor.data) |val| {
            sum += val;
        }
        output_tensor.data[0] = sum / @as(T, @floatFromInt(input_tensor.size));
        return;
    }

    // Get the first axis (we'll focus on single-axis reduction for now)
    var axis: i64 = axes.?[0];
    if (axis < 0) {
        axis += @intCast(input_tensor.shape.len);
    }
    const axis_usize: usize = @intCast(axis);

    // Set up dimensions before and after the reduction axis
    const outer_size: usize = blk: {
        var size: usize = 1;
        for (0..axis_usize) |i| {
            size *= input_tensor.shape[i];
        }
        break :blk size;
    };

    const axis_size: usize = input_tensor.shape[axis_usize];

    const inner_size: usize = blk: {
        var size: usize = 1;
        for (axis_usize + 1..input_tensor.shape.len) |i| {
            size *= input_tensor.shape[i];
        }
        break :blk size;
    };

    // Clear output data
    @memset(output_tensor.data, 0);

    // Perform reduction
    for (0..outer_size) |i| {
        for (0..inner_size) |j| {
            var sum: T = 0;
            for (0..axis_size) |k| {
                // Compute index in flattened input array
                const in_idx = i * axis_size * inner_size + k * inner_size + j;
                sum += input_tensor.data[in_idx];
            }
            // Compute index in flattened output array
            const out_idx = i * inner_size + j;
            output_tensor.data[out_idx] = sum / @as(T, @floatFromInt(axis_size));
        }
    }
}

pub fn get_reduce_mean_output_shape(input_shape: []const usize, axes: ?[]const i64, keepdims: bool, noop_with_empty_axes: bool) ![]usize {
    const allocator = pkg_allocator;

    // Handle empty/null axes case
    if (axes == null or axes.?.len == 0) {
        if (noop_with_empty_axes) {
            // Return copy of input shape
            return try allocator.dupe(usize, input_shape);
        }
        // Reduce over all dimensions
        if (keepdims) {
            const out_shape = try allocator.alloc(usize, input_shape.len);
            @memset(out_shape, 1);
            return out_shape;
        } else {
            var out_shape = try allocator.alloc(usize, 1);
            out_shape[0] = 1;
            return out_shape;
        }
    }

    // Mark dimensions to reduce
    var reduce_dims = try allocator.alloc(bool, input_shape.len);
    defer allocator.free(reduce_dims);
    @memset(reduce_dims, false);

    for (axes.?) |axis| {
        const actual_axis = if (axis < 0)
            @as(usize, @intCast(@as(i64, @intCast(input_shape.len)) + axis))
        else
            @as(usize, @intCast(axis));
        reduce_dims[actual_axis] = true;
    }

    // Count remaining dimensions
    var remaining_dims: usize = 0;
    for (reduce_dims) |is_reduced| {
        if (!is_reduced or keepdims) remaining_dims += 1;
    }

    // Create output shape
    var out_shape = try allocator.alloc(usize, remaining_dims);
    var out_dim: usize = 0;
    for (input_shape, 0..) |dim_size, i| {
        if (!reduce_dims[i] or keepdims) {
            out_shape[out_dim] = if (reduce_dims[i]) 1 else dim_size;
            out_dim += 1;
        }
    }

    return out_shape;
}
