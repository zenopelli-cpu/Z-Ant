const std = @import("std");
const UOp = @import("Uops.zig").UOp;
const DTypeInfo = @import("Uops.zig").DTypeInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp) !void {
    if (uop.op != .MAX and uop.op != .MIN) {
        return error.InvalidOperation;
    }

    //minimal Max and min are only attainable from two variables, not more or less.
    if (uop.src.len != 2) {
        return error.InvalidOperandCount;
    }

    const result_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.id});
    defer allocator.free(result_var);

    const first_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
    defer allocator.free(first_var);

    const type_str = DTypeInfo.asString(uop.dtype);

    switch (uop.op) {
        .EXP2 => try writer.print("const {s} = @as({s}, {s}) * @as({s}, {s});", .{ result_var, type_str, first_var, type_str, first_var}),
        .NEG  => try writer.print("const {s} = @as({s},-{s});",                 .{ result_var, type_str, first_var}),
        else => unreachable,
    }
}


