const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const NodeZant = IR_zant.NodeZant;

// --- utils
pub const utils = IR_zant.utils;
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

    if (codegen_options.log) {
        //log function setting
        try write_logFunction(writer);
    }

    //Fixed Buffer Allocator (only for static allocation)
    if (!codegen_options.dynamic) {
        try write_FBA(writer);
    }

    // _ = linearizedGraph;
    // Generate prediction function code
    try codeGenPredict.writePredict(writer, linearizedGraph, codegen_options.do_export);
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
        \\ 
        \\ // Standard library options for embedded targets
        \\ pub const std_options = std.Options{{
        \\     .page_size_min = 4096,
        \\     .page_size_max = 4096,
        \\     .log_level = .warn,
        \\     .enable_segfault_handler = false,
        \\ }};
        \\ 
        \\ const zant = @import("zant");
        \\ const Tensor = zant.core.tensor.Tensor;
        \\ const tensMath = zant.core.tensor.math_standard;
        \\ const pkgAllocator = zant.utils.allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\ const utils = @import("codegen").codegen_v1.utils;
        \\ const param_lib = @import("static_parameters.zig");
        \\
    , .{});
}

fn write_logFunction(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;
        \\
        \\pub {s} fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {{
        \\    log_function = func;
        \\}}
        \\
    , .{if (codegen_options.do_export == true) "export" else ""});
}

fn write_FBA(writer: std.fs.File.Writer) !void {

    //TODO: instead of hardcoding "buf: [1024 * 10]"" compute the size form the IR Graph
    //
    // Use fixed buffer allocator for static allocations
    try writer.writeAll(
        \\
        \\
        \\ // Static allocation: FixedBufferAllocator
        \\ var buf: [1024 * 10]u8 = undefined;
        \\ var fba_state = std.heap.FixedBufferAllocator.init(&buf);
        \\ const fba = fba_state.allocator();
        \\
    );
}
