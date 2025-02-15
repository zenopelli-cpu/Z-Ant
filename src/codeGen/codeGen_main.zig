const std = @import("std");
const onnx = @import("onnx");
const codeGen = @import("codeGen");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var model1 = try onnx.parseFromFile(allocator, "datasets/models/mnist-1/mnist-1.onnx");
    defer model1.deinit(allocator);

    // onnx.printStructure(&model1);

    const file_path = "src/codeGen/static_lib.zig";
    var file = try std.fs.cwd().createFile(file_path, .{});
    std.debug.print("\n .......... file created, path:{s}", .{file_path});
    defer file.close();

    try codeGen.writeZigFile(file, model1);
}
