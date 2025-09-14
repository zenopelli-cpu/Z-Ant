const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant IR---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;
const IR_utils = IR_zant.utils; //this is IR utils

// --- union ---
const Op_union = @import("../op_union.zig").Op_union;

pub const Fused_Conv_BatchNormalization_Relu = struct {
    op_name: []const u8,
    op_Conv: Op_union,
    op_BatchNormalization: Op_union,
    op_Relu: Op_union,

    pub fn init_fused(fusion_list: std.ArrayList(*NodeZant), op_name: []const u8) !Fused_Conv_BatchNormalization_Relu {
        //Ensure that the ArrayList is the correct one
        if (fusion_list.items.len != 3) return error.WrongNumberOfElements;
        if (!std.mem.eql(u8, fusion_list.items[0].op_type, "Conv")) return error.WrongOpAtPose0;
        if (!std.mem.eql(u8, fusion_list.items[1].op_type, "BatchNormalization")) return error.WrongOpAtPose1;
        if (!std.mem.eql(u8, fusion_list.items[2].op_type, "Relu")) return error.WrongOpAtPose2;

        return Fused_Conv_BatchNormalization_Relu{
            .op_name = op_name,
            .op_Conv = fusion_list.items[0].op,
            .op_BatchNormalization = fusion_list.items[1].op,
            .op_Relu = fusion_list.items[2].op,
        };
    }

    pub fn get_output_shape(self: Fused_Conv_BatchNormalization_Relu) []usize {
        return self.output_C.getShape();
    }

    pub fn get_input_tensors(self: Fused_Conv_BatchNormalization_Relu) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        try input_tensors.append(self.input_A);
        try input_tensors.append(self.input_B);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Fused_Conv_BatchNormalization_Relu) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        try output_tensors.append(self.output_C);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: Fused_Conv_BatchNormalization_Relu, writer: std.fs.File.Writer) !void {
        _ = try writer.print(
            \\
            \\    //{s}
            \\    tensMath.fused_Conv_BatchNormalization_Relu_lean(..., &tensor_{s}) catch return -1;
        , .{
            try IR_utils.getSanitizedName(self.op_name),
            try IR_utils.getSanitizedName(self.op_Relu.output.name), // Output tensor
        });
    }

    pub fn compute_output_shape(self: Fused_Conv_BatchNormalization_Relu) []usize {
        var output_shape: []usize = undefined;
        output_shape = try IR_utils.broadcastShapes(allocator, self.input_A.shape, self.input_B.shape);
        self.output_C.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Fused_Conv_BatchNormalization_Relu) void {
        std.debug.print("\n Fused_Conv_BatchNormalization_Relu:\n {any}", .{self});
    }
};
