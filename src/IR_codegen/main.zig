const std = @import("std");
const zant = @import("zant");
const IR = @import("../IR_graph/IR_graph.zig");
const codegen = @import("IR_codegen.zig");

const onnx = zant.onnx;

const codegen_options = @import("codegen_options");
const codeGen_tests = @import("tests_writer.zig");

// called by "zig build IR_codegen" optionals:" -Dlog -Dmodel="name" -D ..." see build.zig"
pub fn main() !void {
    std.debug.print("\n\ncodegenOptions: ", .{});
    std.debug.print("\n     model:{s} ", .{codegen_options.IR_model});
    std.debug.print("\n     model_path:{s} ", .{codegen_options.IR_model_path});
    std.debug.print("\n     generated_path:{s} ", .{codegen_options.IR_generated_path});
    std.debug.print("\n     user_tests:{s} ", .{codegen_options.IR_user_tests});
    std.debug.print("\n     log:{} ", .{codegen_options.IR_log});
    std.debug.print("\n     shape:{s} ", .{codegen_options.IR_shape});
    std.debug.print("\n     type:{s} ", .{codegen_options.IR_type});
    std.debug.print("\n     output_type:{s} ", .{codegen_options.IR_output_type});
    std.debug.print("\n     comm:{} ", .{codegen_options.IR_comm});
    std.debug.print("\n     dynamic:{} ", .{codegen_options.IR_dynamic});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    const model_name = codegen_options.IR_model;
    const model_path = try std.fmt.allocPrint(gpa_allocator, "datasets/models/{s}/{s}.onnx", .{ model_name, model_name });
    defer gpa_allocator.free(model_path);

    var model = try onnx.parseFromFile(gpa_allocator, model_path);
    defer model.deinit(gpa_allocator);

    //model.print();

    try codegen.codegnenerateFromOnnx(model_name, codegen_options.IR_generated_path, model);

    // Test the generated code
    try codeGen_tests.writeTestFile(model_name, codegen_options.IR_generated_path);
}
