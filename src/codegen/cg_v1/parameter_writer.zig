const std = @import("std");
const zant = @import("zant");
const IR = @import("IR_zant");

// --- zant IR
const GraphZant = IR.GraphZant;
const TensorZant = IR.TensorZant;
// --- utils
pub const utils = @import("utils.zig");
// --- onnx
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
// --- allocator
const allocator = zant.utils.allocator.allocator;
// --- codegen
const codegenParameters = @import("parameters/parameters.zig");

pub fn write(generated_path: []const u8) !void {

    //initializing writer for static_parameters file
    const params_file_path = try std.fmt.allocPrint(allocator, "{s}static_parameters.zig", .{generated_path});
    defer allocator.free(params_file_path);
    var param_file = try std.fs.cwd().createFile(params_file_path, .{});
    std.log.info("\n .......... file created, path:{s}", .{params_file_path});
    defer param_file.close();

    //create writer parameters file
    const writer = param_file.writer();

    // Generate tensor initialization code in the static_parameters.zig file
    try codegenParameters.write_parameters(writer);
}
