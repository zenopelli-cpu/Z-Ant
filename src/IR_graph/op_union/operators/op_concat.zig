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

// https://onnx.ai/onnx/operators/onnx__Concat.html
// INPUTS:
//      - inputs (variadic, heterogeneous) - T: List of tensors for concatenation
// OUTPUTS:
//      - concat_result (heterogeneous) - T: Concatenated tensor
// ATTRIBUTES:
//      - axis (int, required): Which axis to concat on
pub const Concat = struct {
    inputs: std.ArrayList(*TensorZant),
    concat_result: *TensorZant,
    //attributes:
    axis: i64, // default = 1,

    pub fn init(nodeProto: *NodeProto) !Concat {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        const concat_result = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.concat_result_notFound;

        for (nodeProto.input) |input| {
            const ptr = if (tensorZant.tensorMap.getPtr(input)) |ptr| ptr else return error.concat_result_notFound;
            try inputs.append(ptr);
        }
        var axis: i64 = 1.0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != onnx.AttributeType.INT) {
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
            }
        }

        return Concat{
            .inputs = inputs,
            .concat_result = concat_result,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Concat) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn compute_output_shape(self: Concat) []usize {
        var output_shape: []usize = undefined;
        var input_shapes = try allocator.alloc([]const usize, self.inputs.items.len);
        const axis = self.axis;

        for (self.inputs.items, 0..) |input, i| {
            var shape = try allocator.alloc(usize, input.get_shape().len);
            for (input.get_shape(), 0..) |dim, j| {
                shape[j] = if (dim < 0) 1 else @intCast(dim);
            }
            input_shapes[i] = shape;
        }
        output_shape = try tensorMath.get_concatenate_output_shape(input_shapes, axis);
        self.concat_result.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Concat) void { // TODO
        std.debug.print("\n Flatten:\n {any}", .{self});
    }
};
