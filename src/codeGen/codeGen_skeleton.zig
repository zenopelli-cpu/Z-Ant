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
    try write_libraries(writer);

    //Fixed Buffer Allocator
    try write_FBA(writer);

    try write_type_T(writer);

    // Generate tensor initialization code
    try codeGenInitializers.writeTensorsInit(writer, model);

    //try write_debug(writer);

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
inline fn write_libraries(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const Tensor = @import("tensor").Tensor;
        \\ const tensMath = @import("lean_tensor_math");
        \\ const pkgAllocator = @import("pkgAllocator");
        \\ const allocator = pkgAllocator.allocator;
    , .{});
}

fn write_FBA(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\
        \\ var buf: [4096 * 10]f32 = undefined;
        \\ var fba_state = @import("std").heap.FixedBufferAllocator.init(&buf);
        \\ const fba = fba_state.allocator();
    , .{});
}

fn write_type_T(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\
        \\ const T = f32;
    , .{});
}

fn write_debug(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\
        \\export fn ciaoCiao() void {{
        \\      std.debug.print("\n#############################################################", .{{}});
        \\      std.debug.print("\n+                      DEBUG                     +", .{{}});
        \\      std.debug.print("\n#############################################################", .{{}});
        \\}}
    , .{});
}
