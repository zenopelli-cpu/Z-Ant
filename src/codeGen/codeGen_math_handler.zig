const std = @import("std");
const Tensor = @import("tensor").Tensor;
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;
const NodeProto = @import("onnx").NodeProto;
const GraphProto = @import("onnx").GraphProto;
const allocator = @import("pkgAllocator").allocator;
const ReadyNode = @import("codeGen_predict.zig").ReadyNode;

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
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try writer.writeAll("// Handle OneHot\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try writer.writeAll("// Handle Relu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try write_Reshape(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Resize")) {
        try writer.writeAll("// Handle Resize\n");
    } else {
        return error.OperationNotSupported;
    }
}

inline fn write_Reshape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    _ = node;
    try writer.writeAll(
        \\
        \\ TensMath.
    , .{});
}
