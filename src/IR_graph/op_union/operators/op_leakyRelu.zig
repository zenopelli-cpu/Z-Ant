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
const tensorMath = zant.core.tensor.math_standard;

// https://onnx.ai/onnx/operators/onnx__LeakyRelu.html#l-onnx-doc-leakyrelu
// INPUTS:
//      - A (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - C (heterogeneous) - T:  output tensor.
// ATTRIBUTES:
//      - alpha (float) - coefficent of leakage. Default is 0.01.

pub const LeakyRelu = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    alpha: f32 = 0.01, // default value

    pub fn init(nodeProto: *NodeProto) !LeakyRelu {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 0.01; // default value
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "alpha")) {
                alpha = attr.f;
            }
        }

        return LeakyRelu{
            .input_X = input_X,
            .output_Y = output_Y,
            .alpha = alpha,
        };
    }

    pub fn get_output_shape(self: LeakyRelu) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X;
        return res;
    }

    pub fn compute_output_shape(self: LeakyRelu) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_leaky_relu_output_shape(self.input_X.ptr.?.get_shape());
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: LeakyRelu) void {
        std.debug.print("\n LeakyRelu:\n {any}", .{self});
    }
};
