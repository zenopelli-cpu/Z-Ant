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

// https://onnx.ai/onnx/operators/onnx__MatMul.html#l-onnx-doc-matmul// INPUTS:
//      - A (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - C (heterogeneous) - T:  output tensor.
// ATTRIBUTES:
//      - alpha (float) - coefficent of leakage. Default is 0.01.

pub const MatMul = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,
    alpha: f32 = 0.01, // default value

    pub fn init(nodeProto: *NodeProto) !MatMul {
        const input_A = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        var alpha: f32 = 0.01; // default value
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "alpha")) {
                alpha = attr.f;
            }
        }

        return MatMul{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
            .alpha = alpha,
        };
    }

    pub fn get_output_shape(self: MatMul) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X;
        return res;
    }

    pub fn print(self: MatMul) void {
        std.debug.print("\n MatMul:\n {any}", .{self});
    }
};
