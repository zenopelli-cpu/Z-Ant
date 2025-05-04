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

// https://onnx.ai/onnx/operators/onnx__Neg.html#l-onnx-doc-neg
// INPUTS:
//      - X (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - Y (heterogeneous) - T:  output tensor.

pub const Neg = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Neg {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Neg{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Neg) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X;
        return res;
    }

    pub fn print(self: Neg) void {
        std.debug.print("\n Neg:\n {any}", .{self});
    }
};
