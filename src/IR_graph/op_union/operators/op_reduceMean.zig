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

// https://onnx.ai/onnx/operators/onnx__ReduceMean.html
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
//      - axes (optional, heterogeneous) - tensor(int64): A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'keepdims' is true.
// OUTPUTS:
//      - reduced (heterogeneous) - T: Reduced output tensor.
// ATTRIBUTES:
//      - keepdims (int, default is 1): Keep the reduced dimension or not, default 1 means keep the reduced dimension.
//      - noop_with_empty_axes (int, default is 0): Defines behavior if 'axes' is empty. Default behavior is to reduce all axes.
pub const ReduceMean = struct {
    data: *TensorZant,
    axes: ?*TensorZant,
    reduced: *TensorZant,
    //attributes:
    keepdims: bool, // default = true;
    noop_with_empty_axes: bool, // defualt = false;

    pub fn init(nodeProto: *NodeProto) !ReduceMean {
        const data = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const axes = if (nodeProto.input.len > 1) if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.axes_notFound else null;
        const reduced = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var keepdims: bool = true;
        var noop_with_empty_axes: bool = false;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "keepdims")) {
                if (attr.type == onnx.AttributeType.INT) keepdims = attr.i != 0;
            } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
                if (attr.type == onnx.AttributeType.INT) noop_with_empty_axes = attr.i != 0;
            }
        }

        return ReduceMean{
            .data = data,
            .axes = axes,
            .reduced = reduced,
            .keepdims = keepdims,
            .noop_with_empty_axes = noop_with_empty_axes,
        };
    }

    pub fn get_output_shape(self: ReduceMean) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn print(self: ReduceMean) void { // TODO
        std.debug.print("\n ReduceMean:\n {any}", .{self});
    }
};
