const std = @import("std");

const codegen = @import("codegen");
const codegen_options = codegen.codegen_options;

// called by "zig build IR_codegen" optionals:" -Dlog -Dmodel="name" -D ..." see build.zig"
pub fn main() !void {
    std.debug.print("\n\ncodegenOptions: ", .{});
    std.debug.print("\n     model:{s} ", .{codegen_options.model});
    std.debug.print("\n     model_path:{s} ", .{codegen_options.model_path});
    std.debug.print("\n     generated_path:{s} ", .{codegen_options.generated_path});
    std.debug.print("\n     user_tests:{s} ", .{codegen_options.user_tests});
    std.debug.print("\n     log:{} ", .{codegen_options.log});
    std.debug.print("\n     shape:{s} ", .{codegen_options.shape});
    std.debug.print("\n     type:{s} ", .{codegen_options.type});
    std.debug.print("\n     output_type:{s} ", .{codegen_options.output_type});
    std.debug.print("\n     comm:{} ", .{codegen_options.comm});
    std.debug.print("\n     dynamic:{} ", .{codegen_options.dynamic});
    std.debug.print("\n     version:{s} ", .{codegen_options.version});

    if (std.mem.eql(u8, codegen_options.version, "v1")) {
        try codegen.codegen_v1_exe.main();
    }
    // else {
    //     try codegen.codegen_v2_exe.main();
    // }
}
