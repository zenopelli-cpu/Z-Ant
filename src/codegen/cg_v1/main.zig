const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;

const IR_zant = @import("IR_zant");
const IR_graph = IR_zant.IR_graph;
const IR_codegen = @import("codegen_v1.zig");
const codegen_options = @import("codegen_v1.zig").codegen_options;

const codeGen_tests = IR_codegen.testWriter;

// called by "zig build IR_codegen" optionals:" -Dlog -Dmodel="name" -D ..." see build.zig"
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    const model_name = codegen_options.model;
    const model_path = try std.fmt.allocPrint(gpa_allocator, "datasets/models/{s}/{s}.onnx", .{ model_name, model_name });
    defer gpa_allocator.free(model_path);

    var model = try onnx.parseFromFile(gpa_allocator, model_path);
    defer model.deinit(gpa_allocator);

    // model.print(); // << ---------- USEFUL FOR DEBUG

    // try onnx.printModelDetails(&model);  // << ---------- USEFUL FOR DEBUG

    try IR_codegen.codegnenerateFromOnnx(model_name, codegen_options.generated_path, model);

    // Test the generated code
    try codeGen_tests.writeTestFile(model_name, codegen_options.generated_path);
}
