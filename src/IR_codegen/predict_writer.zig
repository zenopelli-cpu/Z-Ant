const std = @import("std");
const zant = @import("zant");
const IR = @import("IR_zant");

// --- zant IR
const GraphZant = IR.GraphZant;
const TensorZant = IR.TensorZant;
const NodeZant = IR.NodeZant;
// --- utils
pub const utils = @import("utils.zig");
// --- onnx
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
// --- allocator
const allocator = zant.utils.allocator.allocator;

// --- codegen
const codeGenPredict = @import("predict/predict.zig");
const codegen_options = @import("codegen_options");

pub fn write(generated_path: []const u8, model_name: []const u8, linearizedGraph: std.ArrayList(*NodeZant)) !void {

    //initializing writer for lib_operation file
    const lib_file_path = try std.fmt.allocPrint(allocator, "{s}lib_{s}.zig", .{ generated_path, model_name });
    defer allocator.free(lib_file_path);
    var lib_file = try std.fs.cwd().createFile(lib_file_path, .{});
    std.log.info("\n .......... file created, path:{s}", .{lib_file_path});
    defer lib_file.close();

    const writer = lib_file.writer();

    // Write the necessary library imports to the generated Zig file
    try write_libraries(writer);

    if (codegen_options.IR_log) {
        //log function setting
        try write_logFunction(writer);
    }

    //Fixed Buffer Allocator
    try write_FBA(writer);

    // _ = linearizedGraph;
    // Generate prediction function code
    try codeGenPredict.writePredict(writer, linearizedGraph, true); //do_export;
}

/// Writes the required library imports to the generated Zig file for predict function.
///
/// This function ensures that the necessary standard and package libraries are
/// imported into the generated Zig source file.
///
/// # Parameters
/// - `writer`: A file writer used to write the import statements.
///
/// # Errors
/// This function may return an error if writing to the file fails.
fn write_libraries(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const zant = @import("zant");
        \\ const Tensor = zant.core.tensor.Tensor;
        \\ const tensMath = zant.core.tensor.math_standard;
        \\ const pkgAllocator = zant.utils.allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\ const codegen = @import("codegen");
        \\ const utils = codegen.utils;
        \\ const param_lib = @import("static_parameters.zig");
        \\
    , .{});
}

fn write_logFunction(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;
        \\
        \\pub export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {{
        \\    log_function = func;
        \\}}
        \\
    , .{});
}

fn write_FBA(writer: std.fs.File.Writer) !void {

    // Use fixed buffer allocator for static allocations
    try writer.writeAll(
        \\
        \\
        \\ // Static allocation: FixedBufferAllocator
        \\ var buf: [4096 * 10]u8 = undefined;
        \\ var fba_state = std.heap.FixedBufferAllocator.init(&buf);
        \\ const fba = fba_state.allocator();
        \\
    );
}
