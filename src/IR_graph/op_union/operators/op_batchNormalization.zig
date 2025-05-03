const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("../../../zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;

// https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
// INPUTS:
//      - X (heterogeneous) - T: Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1
//      - scale (heterogeneous) - T1: Scale tensor of shape ©.
//      - B (heterogeneous) - T1: Bias tensor of shape ©.
//      - input_mean (heterogeneous) - T2: running (training) or estimated (testing) mean tensor of shape ©.
//      - input_var (heterogeneous) - T2: running (training) or estimated (testing) variance tensor of shape ©.
// OUTPUT:
//      - Y (heterogeneous) - T: The output tensor of the same shape as X
// ATTRIBUTES:
//      - epsilon - FLOAT (default is '1e-05'): The epsilon value to use to avoid division by zero.
//      - momentum - FLOAT (default is '0.9'): Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum).
//      - training_mode - INT (default is '0'): If set to true, it indicates BatchNormalization is being used for training, and outputs 1 and 2 are to be computed.

pub const BatchNormalization = struct {
    input_X: *TensorZant,
    scale: *TensorZant,
    B: *TensorZant,
    input_mean: *TensorZant,
    input_var: *TensorZant,
    output_Y: *TensorZant,
    //attributes:
    epsilon: f32, // default = 1e-05;
    momentum: f32, //default = 0.9;
    training_mode: bool, //default = flase;

    pub fn init(nodeProto: *NodeProto) !BatchNormalization {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const scale = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.scale_notFound;
        const B = if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.B_notFound;
        const input_mean = if (tensorZant.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_mean_notFound;
        const input_var = if (tensorZant.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_var_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var epsilon: f32 = 1e-05;
        var momentum: f32 = 0.9;
        const training_mode: bool = false; // -> NOT USED, ALWAYS FALSE for Zant

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "epsilon")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) epsilon = attr.f else return error.BatchNorm_epsilon_NotFloat;
            } else if (std.mem.indexOf(u8, attr.name, "momentum")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) momentum = attr.f else return error.BatchNorm_momentum_NotFloat;
            } else if (std.mem.indexOf(u8, attr.name, "training_mode")) |_| {
                if (attr.type == onnx.AttributeType.INT) if (attr.i != 0) return error.BatchNorm_training_NotAvailable;
            }
        }

        return BatchNormalization{
            .input_X = input_X,
            .scale = scale,
            .B = B,
            .input_mean = input_mean,
            .input_var = input_var,
            .output_Y = output_Y,
            .epsilon = epsilon,
            .momentum = momentum,
            .training_mode = training_mode,
        };
    }
    pub fn get_output_shape(self: BatchNormalization) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn print(self: BatchNormalization) void { // TODO
        std.debug.print("\n BatchNormalization:\n {any}", .{self});
    }
};
