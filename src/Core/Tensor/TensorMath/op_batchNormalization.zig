const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// Onnx standard:
// https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
pub fn batchNormalization(
    comptime T: anytype,
    comptime T1: anytype,
    comptime T2: anytype,
    input: *Tensor(T),
    scales: *Tensor(T1),
    B: *Tensor(T1),
    input_mean: Tensor(T2),
    input_var: Tensor(T2),
    epsilon: f32,
    momentum: f32,
    training_mode: bool,
) !Tensor(T) {

    //checks on the shapes
    if (input.size % scales.size != 0) return error.SizesDontMatch;

    // size(input) % size(shape ©)  == 0

    const output_shape = try get_batchNormalization_output_shape(input);
    var output = Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deint();

    try batchNormalization_lean(T, T1, T2, input, scales, B, input_mean, input_var, epsilon, momentum, training_mode, &output);
}

pub inline fn batchNormalization_lean(
    comptime T: anytype,
    comptime T1: anytype,
    comptime T2: anytype,
    input: *Tensor(T), // X
    scales: *Tensor(T1), //tensor of shape ©.
    B: *Tensor(T1), // tensor of shape ©.
    input_mean: Tensor(T2), // tensor of shape ©.
    input_var: Tensor(T2), // tensor of shape ©.
    epsilon: f32,
    momentum: f32,
    training_mode: bool,
    output: *Tensor(T), // Y
) !void {
    _ = momentum; //reduntant, used only for training
    if (training_mode) {
        std.debug.print("\n\nERROR: training_mode not available for batchNormalization!! \n", .{});
        return error.training_mode_NotAvailable;
    }

    //shape ©
    const c_size = scales.size;
    //number of batches
    const batches = if (input.size % c_size == 0) input.size / c_size else return error.SizesDontMatch;

    //the whole operation
    // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B

    for (0..batches) |i| {
        const stride = i * c_size;

        for (0..c_size) |j| {
            output.data[stride + j] = ((input.data[stride + j] - input_mean.data[j]) / @sqrt(input_var.data[i] + epsilon)) * scales.data[j] + B.data[j];
        }
    }
}

pub inline fn get_batchNormalization_output_shape(comptime T: anytype, input: *Tensor(T)) ![]usize {
    return input.shape;
}
