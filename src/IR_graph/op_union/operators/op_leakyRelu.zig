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

// https://onnx.ai/onnx/operators/onnx__LeakyRelu.html#l-onnx-doc-leakyrelu
// INPUTS:
//      - A (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - C (heterogeneous) - T:  output tensor.
// ATTRIBUTES:
//      - alpha (float) - coefficent of leakage. Default is 0.01.

pub const Mul = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    alpha: f32 = 0.01, // default value

    pub fn init(nodeProto: *NodeProto) !Mul {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 0.01; // default value
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "alpha")) {
                alpha = attr.f;
            }
        }

        return Mul{
            .input_X = input_X,
            .output_Y = output_Y,
            .alpha = alpha,
        };
    }

    pub fn get_output_shape(self: Mul) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X;
        return res;
    }

    pub fn print(self: Mul) void {
        std.debug.print("\n LeakyRelu:\n {any}", .{self});
    }
};
