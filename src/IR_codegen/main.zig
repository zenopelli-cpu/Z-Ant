const std = @import("std");
const zant = @import("zant");
const IR = @import("../IR_graph/IR_graph.zig");
const codegen = @import("IR_codegen.zig");

const onnx = zant.onnx;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    const model_name = "mnist-8";
    const model_path = try std.fmt.allocPrint(gpa_allocator, "datasets/models/{s}/{s}.onnx", .{ model_name, model_name });
    defer gpa_allocator.free(model_path);

    var model = try onnx.parseFromFile(gpa_allocator, model_path);
    defer model.deinit(gpa_allocator);

    model.print();

    try codegen.codegnenerateFromOnnx(model_name, "generated/minst-8/", model);
}
