const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.IR_graph.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.IR_codegen.utils;
pub const Useless = struct {
    pub fn init(nodeProto: *NodeProto) !Useless {
        _ = nodeProto; //"details" will be a onnx struct
        return Useless{};
    }

    pub fn get_output_shape() []usize {
        const res: []usize = [_]usize{ 1, 1, 2, 2 };
        return res;
    }

    pub fn print(self: Useless) void {
        std.debug.print("\n Useless:\n {any}", .{self});
    }

    pub fn get_input_tensors(self: Useless) ![]*TensorZant {
        _ = self;
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Useless) ![]*TensorZant {
        _ = self;
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        return outputs.toOwnedSlice();
    }
};
