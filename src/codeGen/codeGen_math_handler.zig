const std = @import("std");
const Tensor = @import("tensor").Tensor;
const tensorMath = @import("tensor_math");
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const allocator = @import("pkgAllocator").allocator;

// --- proto libs
const TensorProto = @import("onnx").TensorProto;
const NodeProto = @import("onnx").NodeProto;
const GraphProto = @import("onnx").GraphProto;

// --- codeGen libs
const ReadyNode = @import("codeGen_predict.zig").ReadyNode;
const ReadyTensor = @import("codeGen_predict.zig").ReadyTensor;
const utils = @import("codeGen_utils.zig");

/// This method map and write the ONNX operations with the Zant LeanTensorMath mathods
/// Follow the link for details: https://onnx.ai/onnx/operators/?utm_source=chatgpt.com
pub fn write_math_op(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    if (std.mem.eql(u8, node.nodeProto.op_type, "Add")) {
        try writer.writeAll("// Handle Add\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "AveragePool")) {
        try writer.writeAll("// Handle AveragePool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "BatchNormalization")) {
        try writer.writeAll("// Handle BatchNormalization\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Concat")) {
        try writer.writeAll("// Handle Concat\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Constant")) {
        try writer.writeAll("// Handle Constant\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Conv")) {
        try writer.writeAll("// Handle Conv\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Div")) {
        try writer.writeAll("// Handle Div\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Flatten")) {
        try writer.writeAll("// Handle Flatten\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gather")) {
        try writer.writeAll("// Handle Gather\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gemm")) {
        try writer.writeAll("// Handle Gemm\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LeakyRelu")) {
        try writer.writeAll("// Handle LeakyRelu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LogSoftmax")) {
        try writer.writeAll("// Handle LogSoftmax\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MatMul")) {
        try writer.writeAll("// Handle MatMul\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MaxPool")) {
        try writer.writeAll("// Handle MaxPool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mul")) {
        try writer.writeAll("// Handle Mul\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try writer.writeAll("// Handle OneHot\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try writer.writeAll("// Handle Relu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try writer.writeAll("// Handle Relu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Resize")) {
        try writer.writeAll("// Handle Resize\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sigmoid")) {
        try writer.writeAll("// Handle Sigmoid\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Softmax")) {
        try writer.writeAll("// Handle Softmax\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Slice")) {
        try writer.writeAll("// Handle Slice\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Split")) {
        try writer.writeAll("// Handle Split\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sub")) {
        try writer.writeAll("// Handle Sub\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Transpose")) {
        try writer.writeAll("// Handle Transpose\n");
    } else {
        return error.OperationNotSupported;
    }
}

pub fn compute_output_shape(readyNode: *ReadyNode) !void {
    if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Add")) {
        // try writer.writeAll("// Handle Shape Add\n");
        readyNode.outputs.items[0].shape = readyNode.inputs.items[1].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Concat")) {
        // try writer.writeAll("// Handle Shape Concat\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Constant")) {
        // try writer.writeAll("// Handle Shape Constant\n");
        try compute_constant_output(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Conv")) {
        // try writer.writeAll("// Handle Shape Conv\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Div")) {
        // try writer.writeAll("// Handle Shape Div\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Flatten")) {
        // try writer.writeAll("// Handle Shape Flatten\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gather")) {
        // try writer.writeAll("// Handle Shape Gather\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gemm")) {
        // try writer.writeAll("// Handle Shape Gemm\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LeakyRelu")) {
        // try writer.writeAll("// Handle Shape LeakyRelu\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LogSoftmax")) {
        // try writer.writeAll("// Handle Shape LogSoftmax\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MatMul")) {
        // try writer.writeAll("// Handle Shape MatMul\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MaxPool")) {
        // try writer.writeAll("// Handle Shape MaxPool\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mul")) {
        // try writer.writeAll("// Handle Shape Mul\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "OneHot")) {
        // try writer.writeAll("// Handle Shape OneHot\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Relu")) {
        // try writer.writeAll("// Handle Shape Relu\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Reshape")) {
        // try writer.writeAll("// Handle Shape Reshape\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Resize")) {
        // try writer.writeAll("// Handle Shape Resize\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Shape")) {
        // try writer.writeAll("// Handle Shape Shape\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sigmoid")) {
        // try writer.writeAll("// Handle Shape Sigmoid\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Softmax")) {
        // try writer.writeAll("// Handle Shape Softmax\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Slice")) {
        // try writer.writeAll("// Handle Shape Slice\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Split")) {
        // try writer.writeAll("// Handle Shape Split\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sub")) {
        // try writer.writeAll("// Handle Shape Sub\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Transpose")) {
        // try writer.writeAll("// Handle Shape Transpose\n");
    } else {
        std.debug.print("\n\n ERROR! output shape computation for {s} is not available in codeGen_math_handler.compute_output_shape() \n\n", .{readyNode.nodeProto.op_type});
        return error.OperationNotSupported;
    }
}

// ---------------- SHAPE COMPUTATION METHODS ----------------

pub inline fn compute_constant_output(readyNode: *ReadyNode) !void {
    readyNode.outputs.items[0].shape = try utils.getConstantTensorDims(readyNode.nodeProto);
}
