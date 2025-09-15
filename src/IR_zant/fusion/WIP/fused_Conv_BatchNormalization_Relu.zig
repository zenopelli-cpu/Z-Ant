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
const operators = IR_zant.operators;

pub const Fused_Conv_BatchNormalization_Relu = struct {
    op_name: []const u8,
    op_Conv: operators.Conv, // Use the actual Conv type
    op_BatchNormalization: operators.BatchNormalization, // Use the actual BatchNormalization type
    op_Relu: operators.Relu, // Use the actual Relu type

    //inizialization logic for the new operation given a list of old nodes
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Conv_BatchNormalization_Relu {
        //Ensure that the ArrayList is the correct one
        if (fusion_list.items.len != 3) return error.WrongNumberOfElements;
        if (!std.mem.eql(u8, fusion_list.items[0].op_type, "Conv")) return error.WrongOpAtPose0;
        if (!std.mem.eql(u8, fusion_list.items[1].op_type, "BatchNormalization")) return error.WrongOpAtPose1;
        if (!std.mem.eql(u8, fusion_list.items[2].op_type, "Relu")) return error.WrongOpAtPose2;

        // Extract the specific operations from the unions
        const conv_op = switch (fusion_list.items[0].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        const batch_norm_op = switch (fusion_list.items[1].op) {
            .batchNormalization => |bn| bn,
            else => return error.InvalidBatchNormalizationOperation,
        };

        const relu_op = switch (fusion_list.items[2].op) {
            .relu => |r| r,
            else => return error.InvalidReluOperation,
        };

        return Fused_Conv_BatchNormalization_Relu{
            .op_Conv = conv_op,
            .op_BatchNormalization = batch_norm_op,
            .op_Relu = relu_op,
        };
    }

    pub fn get_output_shape(self: Fused_Conv_BatchNormalization_Relu) []usize {
        return self.op_Relu.get_output_shape();
    }

    pub fn get_input_tensors(self: Fused_Conv_BatchNormalization_Relu) anyerror![]*TensorZant {
        return try self.op_Conv.get_input_tensors();
    }

    pub fn get_output_tensors(self: Fused_Conv_BatchNormalization_Relu) anyerror![]*TensorZant {
        return try self.op_Relu.get_output_tensors();
    }

    pub fn write_op(self: Fused_Conv_BatchNormalization_Relu, writer: std.fs.File.Writer) !void {
        _ = try writer.print(
            \\
            \\    //{s}
            \\    tensMath.fused_Conv_BatchNormalization_Relu_lean(..., &tensor_{s}) catch return -1;
        , .{
            "fused Conv-> BatchNormalization -> Relu",
            try IR_utils.getSanitizedName(self.op_Relu.output_Y.name), // Output tensor
        });
    }

    pub fn compute_output_shape(self: Fused_Conv_BatchNormalization_Relu) []usize {
        return self.op_Relu.compute_output_shape();
    }

    pub fn print(self: Fused_Conv_BatchNormalization_Relu) void {
        std.debug.print("\n Fused_Conv_BatchNormalization_Relu:\n {any}", .{self});
    }
};
