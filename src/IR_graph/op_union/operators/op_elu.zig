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

// https://onnx.ai/onnx/operators/onnx__Elu.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
// ATTRIBUTES:
//      - alpha - FLOAT (default is '1.0'): Coefficient of ELU operator
pub const Elu = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    //attributes:
    alpha: f32, // default = 1.0,

    pub fn init(nodeProto: *NodeProto) !Elu {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 1.0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "alpha")) {
                if (attr.type != onnx.AttributeType.FLOAT) {
                    return error.InvalidAttributeType;
                }
                alpha = attr.f;
            }
        }

        return Elu{
            .input_X = input_X,
            .output_Y = output_Y,
            .alpha = alpha,
        };
    }

    pub fn get_output_shape(self: Elu) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn compute_output_shape(self: Elu) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_elu_output_shape(self.input_X.get_shape());
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Elu) void { // TODO
        std.debug.print("\n ADD:\n {any}", .{self});
    }
};
