const std = @import("std");
const zant = @import("zant");
const IR = @import("../IR_graph/IR_graph.zig");
const codegen = @import("IR_codegen.zig");

const onnx = zant.onnx;

const codegen_opt = @import("codegenOptions");

pub fn main() !void {
    std.debug.print("\n\ncodegenOptions: ", .{});
    std.debug.print("\n     model:{s} ", .{codegen_opt.IR_model});
    std.debug.print("\n     model_path:{s} ", .{codegen_opt.IR_model_path});
    std.debug.print("\n     generated_path:{s} ", .{codegen_opt.IR_generated_path});
    std.debug.print("\n     user_tests:{s} ", .{codegen_opt.IR_user_tests});
    std.debug.print("\n     log:{} ", .{codegen_opt.IR_log});
    std.debug.print("\n     shape:{s} ", .{codegen_opt.IR_shape});
    std.debug.print("\n     type:{s} ", .{codegen_opt.IR_type});
    std.debug.print("\n     output_type:{s} ", .{codegen_opt.IR_output_type});
    std.debug.print("\n     comm:{} ", .{codegen_opt.IR_comm});
    std.debug.print("\n     dynamic:{} ", .{codegen_opt.IR_dynamic});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    const model_name = "mnist-8";
    const model_path = try std.fmt.allocPrint(gpa_allocator, "datasets/models/{s}/{s}.onnx", .{ model_name, model_name });
    defer gpa_allocator.free(model_path);

    var model = try onnx.parseFromFile(gpa_allocator, model_path);
    defer model.deinit(gpa_allocator);

    //model.print();

    try codegen.codegnenerateFromOnnx(model_name, "generated/minst-8/", model);
}
