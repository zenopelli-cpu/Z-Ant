const std = @import("std");
const zant = @import("../../../zant.zig");
const allocator = std.heap.page_allocator;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const utils = @import("../../utils.zig");

//https://onnx.ai/onnx/operators/onnx__Sub.html
// INPUTS:
//      - A (heterogeneous) - T: First input tensor
//      - B (heterogeneous) - T: Second input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Sub = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Sub {
        const input_A = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Sub{
            .input_A = input_A,
            .input_B = input_B,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Sub) []usize {
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn compute_output_shape(self: Sub) []usize {
        var output_shape: []usize = undefined;
        output_shape = try utils.broadcastShapes(allocator, self.input_A.shape, self.input_B.shape);
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Sub) void {
        std.debug.print("\n SUB: {any}", .{self});
    }
};
