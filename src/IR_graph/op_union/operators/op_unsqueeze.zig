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
const utils = @import("../../../CodeGen/utils.zig");

//https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
//      - axes (heterogeneous) - T: Axes to unsqueeze
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Unsqueeze = struct {
    input_X: *TensorZant,
    input_axes: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Unsqueeze {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_axes = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_axes_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        return Unsqueeze{
            .input_X = input_X,
            .input_axes = input_axes,
            .output_Y = output_Y,
        };
    }
    pub fn get_output_shape(self: Unsqueeze) []usize { //TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X.shape;
        return res;
    }

    pub fn compute_output_shape(self: Unsqueeze) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_unsqueeze_output_shape(
            self.input_X.shape,
            self.input_axes.ptr.?.i64.data,
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Unsqueeze) void {
        std.debug.print("\n Unsqueeze: {any}", .{self});
    }
};
