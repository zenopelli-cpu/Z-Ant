const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

/// ReLU (Rectified Linear Unit), supporting both standard and quantized tensors.
pub inline fn ReLU_standard(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    // Allocate output with same shape
    var output = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    // Propagate quantization details if present
    output.details = tensor.details;

    try lean_ReLU(T, tensor, &output);

    return output;
}

/// In-place lean ReLU supporting quantized and standard tensors
pub inline fn lean_ReLU(comptime T: anytype, input_tensor: *Tensor(T), output_tensor: *Tensor(T)) !void {

    // Apply ReLU according to tensor type
    switch (input_tensor.details) {
        .quant => |qd| {
            // For quantized tensors: clamp below zero_point
            const zp = qd.zero_point;
            for (0..input_tensor.size) |i| {
                if (input_tensor.data[i] > zp) {
                    output_tensor.data[i] = input_tensor.data[i];
                } else {
                    output_tensor.data[i] = zp;
                }
            }
        },

        else => {
            // Standard numeric ReLU
            for (0..input_tensor.size) |i| {
                if (input_tensor.data[i] > 0) {
                    output_tensor.data[i] = input_tensor.data[i];
                }
            }
        },
    }
}

/// ReLU backward pass, supporting quantized and standard tensors
pub inline fn ReLU_backward(comptime T: anytype, gradient: *Tensor(T), act_relu_input: *Tensor(T)) !void {

    //checks
    if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

    switch (act_relu_input.details) {
        .quant => |qd| {
            // Derivative is 0 when input <= zero_point
            const zp = qd.zero_point;
            for (0..act_relu_input.size) |i| {
                if (act_relu_input.data[i] <= zp)
                    gradient.data[i] = 0;
                // else gradient remains unchanged since f'(x) = 1 for x > zero_point
            }
        },

        else => {
            // Standard ReLU derivative: 0 if x <= 0
            for (0..act_relu_input.size) |i| {
                if (act_relu_input.data[i] <= 0)
                    gradient.data[i] = 0;
                // else gradient remains unchanged since f'(x) = 1 for x > 0
            }
        },
    }
}
