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

//https://onnx.ai/onnx/operators/onnx__Split.html
// INPUTS:
//      - input (heterogeneous) - T: Input tensor
//      - split (optional, heterogeneous) - tensor(int64):
// OUTPUTS:
//      - output (heterogeneous) - T: Output tensor
// ATTRIBUTES:
//      - axis - INT (default is '0'): Indicate up to which input dimension should be split.
//      - num_outputs - INT: Number of outputs
pub const Split = struct {
    input: *TensorZant,
    split: ?*TensorZant,
    output_Y: *TensorZant,
    //attributes:
    axis: i64 = 0, // default = 0,

    pub fn init(nodeProto: *NodeProto) !Split {
        const input = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const splitTensor = if (nodeProto.input.len > 1) if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.axes_notFound else null;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = 0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i;
            }
        }

        return Split{
            .input = input,
            .split = splitTensor,
            .output_Y = output_Y,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Split) []usize {
        return self.output_Y.shape;
    }

    pub fn print(self: Split) void {
        std.debug.print("\n Split: {any}", .{self});
    }
};
