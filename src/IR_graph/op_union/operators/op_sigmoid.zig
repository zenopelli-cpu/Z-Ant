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

//https://onnx.ai/onnx/operators/onnx__Sigmoid.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Sigmoid = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Sigmoid {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Sigmoid{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Sigmoid) []usize {
        return self.output_Y.shape;
    }

    pub fn print(self: Sigmoid) void {
        std.debug.print("\n Sigmoid: {any}", .{self});
    }
};
