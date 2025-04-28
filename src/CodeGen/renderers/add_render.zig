const std = @import("std");
const UOp = @import("Uops.zig").UOp;

pub fn render(writer: std.fs.File.Writer, uop: UOp) !void {
    try std.fmt.format(writer, "add {d}\n", .{uop.id});
}
