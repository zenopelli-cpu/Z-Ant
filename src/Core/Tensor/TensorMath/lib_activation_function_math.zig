const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const TensorError = @import("errorHandler").TensorError;
const Converter = @import("typeC");
const ArchitectureError = @import("errorHandler").ArchitectureError;

/// ReLU (Rectified Linear Unit).
/// It outputs the input directly if it's positive, but returns zero for any negative input.
pub inline fn ReLU_standard(comptime T: anytype, tensor: *Tensor(T)) !void {
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;
    try lean_ReLU(T, tensor);
}

pub inline fn lean_ReLU(comptime T: anytype, tensor: *Tensor(T)) !void {
    //apply ReLU
    //OSS: can be improved, see how did I parallelized CPU Tensor Sum
    for (0..tensor.size) |i| {
        if (tensor.data[i] <= 0) tensor.data[i] = 0;
    }
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
pub inline fn leakyReLU(comptime T: anytype, tensor: *Tensor(T), slope: T) !void {
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    try lean_leakyReLU(T, tensor, slope);
}

pub inline fn lean_leakyReLU(comptime T: anytype, tensor: *Tensor(T), slope: T) !void {
    //apply Leaky ReLU suing relu self.relu() - (-neg_slope*self).relu()
    for (0..tensor.size) |i| {
        if (tensor.data[i] <= 0) {
            tensor.data[i] = slope * tensor.data[i];
        }
    }
}

/// The Sigmoid activation function is a smooth, S-shaped function that maps any input
/// to a value between 0 and 1.
/// it can suffer from vanishing gradients, especially for large positive or negative
/// inputs, slowing down training in deep networks.
pub inline fn sigmoid(comptime T: anytype, tensor: *Tensor(T)) !void {
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    try sigmoid_lean(T, tensor);
}

pub inline fn sigmoid_lean(comptime T: anytype, tensor: *Tensor(T)) !void {
    //apply Sigmoid
    for (0..tensor.size) |i| {
        tensor.data[i] = 1.0 / (1.0 + @exp(-tensor.data[i]));
    }
}

/// The Softmax activation function is used in multi-class classification tasks to convert
/// logits (raw output values) into probabilities that sum to 1.
/// Ideal for output layers in multi-class neural networks.
pub fn softmax(comptime T: anytype, tensor: *Tensor(T)) !void {

    // TODO: checks

    //compute
    try lean_softmax(T, tensor);
}

pub inline fn lean_softmax(comptime T: anytype, input: *Tensor(T)) !void {
    const rows = input.shape[0];
    const cols = input.shape[1];

    var max_val: T = undefined;
    var sum_of_exp: T = 0.0;
    var val: T = undefined;

    // For each row
    for (0..rows) |i| {
        // Find the maximum value in the row to stabilize the computation
        max_val = input.data[i * cols];
        for (0..cols) |j| {
            val = input.data[i * cols + j];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute stabilized exponentials and their sum
        sum_of_exp = 0.0;
        for (0..cols) |j| {
            val = input.data[i * cols + j] - max_val; // Stabilization
            val = @exp(val);
            input.data[i * cols + j] = val;
            sum_of_exp += val;
        }

        // Normalize to calculate the softmax
        for (0..cols) |j| {
            input.data[i * cols + j] /= sum_of_exp;
        }
    }
}

// //TODO: now scan the rows of the matrix, it must scan the columns
// fn compute_mutidim_softmax(input: *Tensor(T), current_depth: usize, location: []usize) !void {
//     if (current_depth == (input.shape.len - 1)) {
//         //declaring res as the result of the sum of the MSE
//         const allocator = pkg_allocator;

//         //get location is used just to manage the gets and sets relative to the current depth
//         const get_location = try allocator.alloc(usize, location.len);
//         defer allocator.free(get_location);
//         //initializing get location to the same values of location
//         for (0..get_location.len) |i| {
//             get_location[i] = location[i];
//         }

//         //input.info();

//         //allocating space for the exponent of each value
//         var sum_of_exp: T = 0.0;
//         var val: T = undefined;
//         var exp: T = undefined;

//         //calculating the value of the exponential for each element
//         for (0..input.shape[current_depth]) |i| {
//             get_location[current_depth] = i; //for each element of predictions vect and target vect
//             val = try input.get_at(get_location);
//             exp = @exp(val);
//             try input.set_at(get_location, exp);
//             sum_of_exp += exp;
//         }

//         //set the value of current_elem/sum_of_exp
//         for (0..input.shape[current_depth]) |i| {
//             get_location[current_depth] = i; //for each element of predictions vect and target vect
//             val = try input.get_at(get_location);
//             val = val / sum_of_exp;
//             try input.set_at(get_location, val);
//         }
//     } else {
//         for (0..input.shape[current_depth]) |element_at_current_depth| {
//             //print depth:
//             //std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
//             location[current_depth] = element_at_current_depth;
//             //otherwise I have to go deeper
//             try compute_mutidim_softmax(
//                 input,
//                 current_depth + 1,
//                 location,
//             );
//         }
//     }
// }
