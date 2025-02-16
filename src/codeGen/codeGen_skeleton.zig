const std = @import("std");
const Tensor = @import("tensor").Tensor;
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;
const allocator = @import("pkgAllocator").allocator;
const codeGenInitializers = @import("codeGen_initializers.zig");
const coddeGenPredict = @import("codeGen_predict.zig");
const tensorMath = @import("tensor_math");

/// Writes a Zig source file containing the generated code for an ONNX model.
///
/// This function generates the necessary Zig code to initialize tensors and
/// define the prediction logic based on the given ONNX model.
///
/// # Parameters
/// - `file`: The file where the generated Zig code will be written.
/// - `model`: The ONNX model from which to generate the Zig code.
///
/// # Errors
/// This function may return an error if writing to the file fails.
pub fn writeZigFile(file: std.fs.File, model: ModelOnnx) !void {
    const writer = file.writer();

    // Write the necessary library imports to the generated Zig file
    try writeLibraries(writer);

    // Generate tensor initialization code
    try codeGenInitializers.writeTensorsInit(writer, model);

    // Generate prediction function code
    try coddeGenPredict.writePredict(writer, model);
}

/// Writes the required library imports to the generated Zig file.
///
/// This function ensures that the necessary standard and package libraries are
/// imported into the generated Zig source file.
///
/// # Parameters
/// - `writer`: A file writer used to write the import statements.
///
/// # Errors
/// This function may return an error if writing to the file fails.
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
