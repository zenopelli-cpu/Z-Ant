const std = @import("std");
const Tensor = @import("tensor").Tensor;

pub const NormalizationType = enum {
    null,
    UnityBasedNormalizartion,
    StandardDeviationNormalization,
};

pub fn normalize(comptime T: anytype, tensor: *Tensor(T), normalizationType: NormalizationType) !void {
    switch (normalizationType) {
        NormalizationType.UnityBasedNormalizartion => try multidimNormalizeUnityBased(T, tensor),
        NormalizationType.StandardDeviationNormalization => try multidimNormalizeStandard(T, tensor),
        else => try multidimNormalizeUnityBased(T, tensor),
    }
}

/// Normalize each row in a multidimensional tensor
fn multidimNormalizeUnityBased(comptime T: anytype, tensor: *Tensor(T)) !void {
    // --- Checks ---
    // T must be float
    if (@typeInfo(T) != .Float) return error.NotFloatType;

    var counter: usize = 0; //counter counts the rows in all the tensor, indipendently of the shape
    const cols: usize = tensor.shape[tensor.shape.len - 1]; //aka: elements per row
    const numb_of_rows = tensor.data.len / cols;

    var delta: T = 0;

    while (counter < numb_of_rows) {
        var max = tensor.data[counter * cols];
        var min = tensor.data[counter * cols];

        // Find max and min for each row
        for (0..cols) |i| {
            if (tensor.data[counter * cols + i] > max) max = tensor.data[counter * cols + i];
            if (tensor.data[counter * cols + i] < min) min = tensor.data[counter * cols + i];
        }
        delta = max - min;

        // Update tensor for 1D normalization
        for (0..cols) |i| {
            tensor.data[counter * cols + i] = if (delta == 0) 0 else (tensor.data[counter * cols + i] - min) / delta;
        }

        counter += 1;
    }
}

fn multidimNormalizeStandard(comptime T: anytype, tensor: *Tensor(T)) !void {
    std.debug.print("\n     standard normalize ", .{});
    // --- Type Checks ---
    // Ensure that T is a floating-point type
    if (@typeInfo(T) != .Float) return error.NotFloatType;

    // --- Shape Checks ---
    // Expecting a tensor with at least 2 dimensions
    if (tensor.shape.len < 2) {
        return error.InvalidTensorShape;
    }

    const last_dim_size = tensor.shape[tensor.shape.len - 1];
    var num_slices: usize = 1;
    for (tensor.shape[0 .. tensor.shape.len - 1]) |dim| {
        num_slices *= dim;
    }

    // Small epsilon value to prevent division by zero
    const epsilon = @as(T, 1e-7);

    // Iterate over each slice
    for (0..num_slices) |slice| {
        const slice_offset = slice * last_dim_size;

        // --- Step 1: Compute Mean ---
        var sum: T = 0.0;
        for (0..last_dim_size) |i| {
            sum += tensor.data[slice_offset + i];
        }
        const mean = sum / @as(T, @floatFromInt(last_dim_size));

        // --- Step 2: Compute Variance ---
        var sum_sq_diff: T = 0.0;
        for (0..last_dim_size) |i| {
            const diff: T = tensor.data[slice_offset + i] - mean;
            sum_sq_diff += diff * diff;
        }
        const variance = sum_sq_diff / @as(T, @floatFromInt(last_dim_size));

        // --- Step 3: Compute Standard Deviation ---
        const std_dev = std.math.sqrt(variance + epsilon);

        // --- Step 4: Normalize Each Element ---
        for (0..last_dim_size) |i| {
            if (std_dev > 0) {
                tensor.data[slice_offset + i] = (tensor.data[slice_offset + i] - mean) / std_dev;
            } else {
                // If standard deviation is zero, set normalized value to 0
                tensor.data[slice_offset + i] = 0;
            }
        }
    }
}
