const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen").codegen_v2;
const utils = codegen.utils;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const TensorProto = onnx.TensorProto;
const DataType = onnx.DataType;
const globals = codegen.globals;
const IR_graph = zant.IR_graph;
const TensorLib = IR_graph.tensorZant_lib;
const IR_graph_utils = IR_graph.utils;
const TensorZant = IR_graph.TensorZant;

const allocator = zant.utils.allocator.allocator;

const Writer = std.fs.File.Writer;

/// Configuration for tensor code generation
const TensorConfig = struct {
    const header =
        \\// ---------------------------------------------------
        \\// +         Initializing Weights and Biases         +
        \\// ---------------------------------------------------
    ;

    const tensor_comment = "// ----------- Initializing tensor_{s};";
    const shape_template = "const shape_tensor_{s} : [{}]usize = [_]usize{{ ";
    const stride_template = "const stride_tensor_{s} : [{}]usize = [_]usize{{ ";
    const array_template = "const array_{s} : [{d}]{s} linksection(\".rodata\") = [_]{s}{{";
};

/// Writes the Zig code required to initialize all tensor initializers in the ONNX model.
pub fn write_parameters(writer: Writer) !void {
    try writer.print(TensorConfig.header, .{});

    const initializers = try IR_graph_utils.getInitializers(&TensorLib.tensorMap);
    for (initializers) |*tensor| {
        try writeTensorInitializer(writer, tensor);
    }
}

/// Writes a complete tensor initializer (shape, stride, and data arrays).
fn writeTensorInitializer(writer: Writer, tensor: *TensorZant) !void {
    const name = try utils.getSanitizedName(tensor.name);
    defer allocator.free(name);

    try writer.print("\n\n" ++ TensorConfig.tensor_comment, .{name});
    try writeTensorShape(writer, tensor, name);
    try writeTensorStride(writer, tensor, name);
    try writeTensorData(writer, tensor, name);
}

/// Writes the shape array for a tensor.
fn writeTensorShape(writer: Writer, tensor: *TensorZant, name: []const u8) !void {
    const shape = tensor.getShape();
    try writer.print("\n\n" ++ TensorConfig.shape_template, .{ name, shape.len });
    try writeIntegerArray(writer, shape);
    try writer.print(" }};", .{});
}

/// Writes the stride array for a tensor.
fn writeTensorStride(writer: Writer, tensor: *TensorZant, name: []const u8) !void {
    const stride = tensor.getStride();
    try writer.print("\n\n" ++ TensorConfig.stride_template, .{ name, stride.len });
    try writeIntegerArray(writer, stride);
    try writer.print(" }};", .{});
}

/// Writes the data array for a tensor.
fn writeTensorData(writer: Writer, tensor: *TensorZant, name: []const u8) !void {
    std.log.info("\n[writeArray] Processing tensor: {s}, DataType: {s}", .{ name, tensor.ty.toString() });

    const data_type = tensor.ty.toString();
    const size = calculateTensorSize(tensor.shape);

    try writer.print("\n" ++ TensorConfig.array_template, .{ name, size, data_type, data_type });

    if (tensor.ptr) |tensor_data| {
        try writeTensorBytes(writer, tensor_data.get_data_bytes());
    }

    try writer.print(" }};", .{});
}

/// Calculates the total size of a tensor from its shape.
fn calculateTensorSize(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }
    return size;
}

/// Writes an array of integers with comma separation.
fn writeIntegerArray(writer: Writer, data: []const usize) !void {
    for (data, 0..) |value, i| {
        if (i > 0) try writer.print(", ", .{});
        try writer.print(" {}", .{value});
    }
}

/// Writes tensor bytes as comma-separated values.
fn writeTensorBytes(writer: Writer, data: []const u8) !void {
    for (data, 0..) |byte, i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print(" {d}", .{byte});
    }
}

/// Generic function to write array data of any type.
pub fn writeArrayData(writer: Writer, comptime T: type, data: []const T) !void {
    for (data, 0..) |value, i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print(" {}", .{value});
    }
}
