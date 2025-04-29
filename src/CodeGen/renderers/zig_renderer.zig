const std = @import("std");

const UOp = @import("Uops.zig").UOp;
const UOpType = @import("Uops.zig").UOpType;

const AddRender = @import("add_render.zig");

pub const ZigRenderer = struct {
    allocator: std.mem.Allocator,
    writer: std.fs.File.Writer,

    pub fn init(allocator: std.mem.Allocator, writer: std.fs.File.Writer) @This() {
        return .{
            .allocator = allocator,
            .writer = writer,
        };
    }

    pub fn deinit(_: @This()) void {
        // Cleanup if needed
    }

    pub fn render(self: *@This(), uops: []const UOp) !void {
        for (uops) |uop| {
            switch (uop.op) {
                .ADD => try AddRender.render(self.writer, uop),
                else => {
                    try std.fmt.format(self.writer, "unknown op {d}\n", .{uop.id});
                },
            }
        }
    }
};

test "ZigRenderer" {
    const allocator = std.testing.allocator;
    const file = try std.fs.cwd().createFile("test_output.zig", .{});
    defer file.close();

    const writer = file.writer();
    var renderer = ZigRenderer.init(allocator, writer);
    defer renderer.deinit();

    const uops = &.{
        UOp{ .id = 0, .op = .ADD, .dtype = .f32, .src = &[_]usize{ 1, 2 }, .arg = null },
        UOp{ .id = 1, .op = .ADD, .dtype = .f32, .src = &[_]usize{ 3, 4 }, .arg = null },
    };

    try renderer.render(uops);
}
