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

// https://onnx.ai/onnx/operators/onnx__Identity.html#l-onnx-doc-identity
// INPUTS:
//      - input (heterogeneous) - V:  input tensor.
// OUTPUTS:
//      - output (heterogeneous) - V:  output tensor.

pub const Identity = struct {
    input: *TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Identity {
        const input = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_notFound;
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        return Identity{
            .input = input,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Identity) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X;
        return res;
    }

    pub fn compute_output_shape(self: Identity) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_identity_output_shape(self.input.ptr.?.get_shape());
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Identity) void {
        std.debug.print("\n Identity:\n {any}", .{self});
    }
};
