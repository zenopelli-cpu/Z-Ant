const std = @import("std");
const Tensor = @import("tensor").Tensor;
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;
const allocator = @import("pkgAllocator").allocator;
const codeGenInitializers = @import("codeGen_initializers.zig");
const coddeGenPredict = @import("codeGen_predict.zig");

pub fn writeZigFile(file: std.fs.File, model: ModelOnnx) !void {
    const writer = file.writer();

    try writeLibraries(writer);

    try codeGenInitializers.writeTensorsInit(writer, model);

    try coddeGenPredict.writePredict(writer, model);
}

inline fn writeLibraries(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const Tensor = @import("tensor").Tensor;
        \\ const TensMath = @import("lean_tensor_math");
        \\ const pkgAllocator = @import("pkgAllocator");
        \\ const allocator = pkgAllocator.allocator;
    , .{});
}
