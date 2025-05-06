const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("../../../zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const tensorMath = zant.core.tensor.math_standard;

//https://onnx.ai/onnx/operators/onnx__Tanh.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Tanh = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Tanh {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Tanh{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Tanh) []usize {
        return self.output_Y.shape;
    }

    pub fn print(self: Tanh) void {
        std.debug.print("\n Tanh: {any}", .{self});
    }

    pub fn compute_output_shape(self: Tanh) ![]usize {
        const shape: []const usize = undefined;
        const input_shape = self.input_X.get_shape();
        shape = try tensorMath.get_tanh_output_shape(input_shape);
        return shape;
    }
};
