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

// https://onnx.ai/onnx/operators/onnx__Reshape.html#l-onnx-doc-reshape
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
//      - shape (heterogeneous) - tensor(int64): Specified shape for output
// OUTPUTS:
//      - reshaped (heterogeneous) - T: Reshaped data.
// ATTRIBUTES:
//      - allowzero - INT (default is '0'): If '1', the shape can contain zero. TODO

pub const Reshape = struct {
    data: *TensorZant,
    shape: *TensorZant,
    reshaped: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Reshape {
        const data = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const shape = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.shape_notFound;
        const reshaped = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Reshape{
            .data = data,
            .shape = shape,
            .reshaped = reshaped,
        };
    }

    pub fn get_output_shape(self: Reshape) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X;
        return res;
    }

    pub fn print(self: Reshape) void {
        std.debug.print("\n Reshape:\n {any}", .{self});
    }
};
