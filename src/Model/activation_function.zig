const std = @import("std");
const Tensor = @import("tensor").Tensor;
//import error libraries
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;
const TensMath = @import("tensor_m");

pub const ActivationType = enum {
    ReLU,
    Sigmoid,
    Softmax,
    None,
    LeakyReLU,
};

/// Activation function Interface, used to instantiate a Loss Function struct
/// depending on the ActivationType passed by argument.
pub fn ActivationFunction(comptime T: anytype, activationType: ActivationType) type {
    return switch (activationType) {
        ActivationType.ReLU => ReLU(T),
        ActivationType.Sigmoid => Sigmoid(T),
        ActivationType.Softmax => Softmax(T),
        ActivationType.LeakyReLU => LeakyReLU(T),
        ActivationType.None => None(),
    };
}

/// Used when no activation functio is needed
pub fn None() type {}

/// ReLU (Rectified Linear Unit).
/// It outputs the input directly if it's positive, but returns zero for any negative input.
pub fn ReLU(comptime T: anytype) type {
    return struct {
        const Self = @This();

        //it directly modify the input tensor
        //threshold is usually set to zero
        pub fn forward(self: *Self, input: *Tensor(T)) !void {
            _ = self;

            try TensMath.ReLU(T, input);
        }

        pub fn derivate(self: *Self, gradient: *Tensor(T), act_relu_input: *Tensor(T)) !void {
            _ = self;
            //checks
            if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
            if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

            //apply ReLU derivative: f'(x) = 0 if x <= 0, 1 if x > 0
            for (0..gradient.size) |i| {
                if (act_relu_input.data[i] <= 0) {
                    gradient.data[i] = 0;
                }
                // else gradient remains unchanged since f'(x) = 1 for x > 0
            }
        }
    };
}
/// Leaky ReLU is a variant of ReLU that allows a small, positive gradient when the input is negative.
/// This can help prevent the dying ReLU problem, where neurons stop learning if they get stuck in the negative side of the ReLU function.
/// The leaky parameter is a small value that determines how much the function leaks into the negative side.
/// A common value for the leaky parameter is 0.01.
/// The Leaky ReLU function is defined as:
/// f(x) = x if x > 0
/// f(x) = alpha * x if x <= 0
/// where alpha is a small constant.
/// The derivative of the Leaky ReLU function is:
/// f'(x) = 1 if x > 0
/// f'(x) = alpha if x <= 0
pub fn LeakyReLU(comptime T: anytype) type {
    return struct {
        const Self = @This();
        pub fn forward(self: *Self, input: *Tensor(T), slope: T) !void {
            _ = self;

            try TensMath.leakyReLU(T, input, slope);
        }

        pub fn derivate(self: *Self, gradient: *Tensor(T), act_relu_input: *Tensor(T), slope: T) !void {
            _ = self;
            //checks
            if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
            if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

            //apply Leaky ReLU derivative: f'(x) = 1 if x > 0, slope if x <= 0
            for (0..gradient.size) |i| {
                gradient.data[i] *= if (act_relu_input.data[i] > 0) 1 else slope;
            }
        }
    };
}

/// The Sigmoid activation function is a smooth, S-shaped function that maps any input
/// to a value between 0 and 1.
/// it can suffer from vanishing gradients, especially for large positive or negative
/// inputs, slowing down training in deep networks.
pub fn Sigmoid(comptime T: anytype) type {
    return struct {
        const Self = @This();
        //it directly modify the input tensor
        pub fn forward(self: *Self, input: *Tensor(T)) !void {
            _ = self;
            try TensMath.sigmoid(T, input);
        }

        pub fn derivate(self: *Self, gradient: *Tensor(T), act_forward_out: *Tensor(T)) !void {
            _ = self;
            //checks
            if (gradient.size <= 0 or act_forward_out.size <= 0) return TensorError.ZeroSizeTensor;
            if (gradient.size != act_forward_out.size) return TensorMathError.InputTensorDifferentSize;

            //apply Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
            for (0..gradient.size) |i| {
                const sigmoid_output = act_forward_out.data[i];
                gradient.data[i] *= sigmoid_output * (1 - sigmoid_output);
            }
        }
    };
}

const pkg_allocator = @import("pkgAllocator").allocator;

/// The Softmax activation function is used in multi-class classification tasks to convert
/// logits (raw output values) into probabilities that sum to 1.
/// Ideal for output layers in multi-class neural networks.
pub fn Softmax(comptime T: anytype) type {
    return struct {
        const Self = @This();
        //it directly modify the input tensor
        pub fn forward(self: *Self, input: *Tensor(T)) !void {
            _ = self;

            //try compute_mutidim_softmax(input, 0, location);
            try TensMath.softmax(T, input);
        }

        pub fn derivate(self: *Self, dL_dX: *Tensor(T), softmax_output: *Tensor(T)) !void {
            _ = self;
            //checks
            if (dL_dX.size <= 0) return TensorError.ZeroSizeTensor;
            if (dL_dX.size != softmax_output.size) return TensorMathError.InputTensorDifferentSize;

            const dim = dL_dX.shape.len;
            const rows = dL_dX.shape[dim - 2];
            const cols = dL_dX.shape[dim - 1];

            // Make a copy of incoming gradients since we'll modify dL_dX
            var dL_dS = try dL_dX.copy();
            defer dL_dS.deinit();

            // For each sample in the batch
            for (0..rows) |i| {
                const row_offset = i * cols;
                // For each output neuron j
                for (0..cols) |j| {
                    var sum: T = 0;
                    const sj = softmax_output.data[row_offset + j];

                    // Compute sum of dL/dS_k * S_k for all k
                    for (0..cols) |k| {
                        const sk = softmax_output.data[row_offset + k];
                        const dL_dS_k = dL_dS.data[row_offset + k];
                        if (j == k) {
                            sum += dL_dS_k * sk * (1 - sj); // Diagonal term
                        } else {
                            sum += dL_dS_k * (-sk * sj); // Off-diagonal term
                        }
                    }
                    dL_dX.data[row_offset + j] = sum;
                }
            }
        }
    };
}
