const std = @import("std");
const UOp = @import("Uops.zig").UOp;
const DTypeInfo = @import("Uops.zig").DTypeInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp) !void {
    if (uop.op != .REDUCE_ADD and uop.op != .REDUCE_MAX) {
        return error.InvalidOperation;
    }

    if (uop.src.len != 1) {
        return error.InvalidOperandCount;
    }

    const result_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.id});
    defer allocator.free(result_var);

    const input_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
    defer allocator.free(input_var);

    const type_str = DTypeInfo.asString(uop.dtype);

    // Vector size
    const vec_size = try std.mem.join(allocator, ".", &.{ input_var, "len" });
    defer allocator.free(vec_size);
    // Vector definition
    try writer.print(
        \\const vec{d}: @Vector({s}, {s}) = {s};
        \\
    , .{ uop.id, vec_size, type_str, input_var });

    // Operation mapping
    const op_str = switch (uop.op) {
        .REDUCE_ADD => ".Add",
        .REDUCE_MAX => ".Max",
        else => unreachable,
    };

    // Final reduction line
    try writer.print("const {s}: {s} = @reduce({s}, vec{d});\n", .{ result_var, type_str, op_str, uop.id });
}
