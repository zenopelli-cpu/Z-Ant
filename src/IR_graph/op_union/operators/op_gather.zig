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
const utils = @import("../../../CodeGen/utils.zig");

//https://onnx.ai/onnx/operators/onnx__Greater.html#l-onnx-doc-greater
// INPUTS:
//      - A (heterogeneous) - T:  First input operand for the logical operator.
//      - B (heterogeneous) - T:  Second input operand for the logical operator.
// OUTPUTS:
//      - C (heterogeneous) - T:  result tensor.

pub const Gather = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Gather {
        const input_A = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        return Gather{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
        };
    }
    pub fn get_output_shape(self: Gather) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_A;
        return res;
    }

    // pub fn compute_output_shape(self: Gather) []usize {
    //     var output_shape: []usize = undefined;

    //     output_shape = try utils.usizeSliceToI64Slice(try tensorMath.get_gather_output_shape(
    //         try utils.i64SliceToUsizeSlice(data_shape),
    //         try utils.i64SliceToUsizeSlice(indices_shape),
    //         axis,
    //     ));
    // }
    pub fn print(self: Gather) void { //TODO
        std.debug.print("\n Gather:\n {any}", .{self});
    }
};
