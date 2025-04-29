const std = @import("std");
const UOp = @import("Uops.zig").UOp;
const UOpType = @import("Uops.zig").UOpType;

const ArithmeticRender = @import("arithmetic_render.zig");

pub fn ZigRenderer(comptime WriterType: type) type {
    return struct {
        allocator: std.mem.Allocator,
        writer: WriterType,

        pub fn init(allocator: std.mem.Allocator, writer: WriterType) @This() {
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
                    .ADD, .SUB, .MUL, .FDIV => try ArithmeticRender.render(self.allocator, self.writer, uop),
                    else => {
                        try std.fmt.format(self.writer, "unknown op {d}\n", .{uop.id});
                    },
                }
            }
        }
    };
}

test "Arithemtic operations" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    const uops = &.{
        UOp{ .id = 0, .op = .ADD, .dtype = .f32, .src = &[_]usize{ 1, 2 }, .arg = null },
        UOp{ .id = 1, .op = .SUB, .dtype = .f32, .src = &[_]usize{ 3, 4 }, .arg = null },
        UOp{ .id = 2, .op = .MUL, .dtype = .f32, .src = &[_]usize{ 5, 6 }, .arg = null },
        UOp{ .id = 3, .op = .FDIV, .dtype = .f32, .src = &[_]usize{ 7, 8 }, .arg = null },
    };

    try renderer.render(uops);

    const expected =
        \\const t0 = @as(f32, t1) + @as(f32, t2);
        \\const t1 = @as(f32, t3) - @as(f32, t4);
        \\const t2 = @as(f32, t5) * @as(f32, t6);
        \\const t3 = @as(f32, t7) / @as(f32, t8);
        \\
    ;
    const actual = try buffer.toOwnedSlice();
    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}
