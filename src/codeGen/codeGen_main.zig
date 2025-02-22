const std = @import("std");

const onnx = @import("onnx");
const Tensor = @import("tensor").Tensor;
const tensorMath = @import("tensor_math");
const allocator = @import("pkgAllocator").allocator;
const codeGen = @import("codeGen_skeleton.zig");
const codeGen_utils = @import("codeGen_utils.zig");
const codeGen_init = @import("codeGen_initializers.zig");
const codeGen_mathHandl = @import("codeGen_math_handler.zig");
const codeGen_predict = @import("codeGen_predict.zig");

const codegen_options = @import("codegen_options");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    var model1 = try onnx.parseFromFile(gpa_allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model1.deinit(gpa_allocator);

    //onnx.printStructure(&model1);

    const file_path = "src/codeGen/static_lib.zig";
    var file = try std.fs.cwd().createFile(file_path, .{});
    std.debug.print("\n .......... file created, path:{s}", .{file_path});
    defer file.close();

    try codeGen.writeZigFile(file, model1);
}
