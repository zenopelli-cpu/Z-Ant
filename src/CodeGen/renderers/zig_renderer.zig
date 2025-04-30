const std = @import("std");
const UOp = @import("Uops.zig").UOp;
const UOpType = @import("Uops.zig").UOpType;

const ArithmeticRender = @import("arithmetic_render.zig");
const ReduceRender = @import("reduce_render.zig");
const ConditionalRender = @import("conditional_render.zig");
const UnaryRender = @import("unary_render.zig");

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

        pub fn render(self: *@This(), uops: []UOp) !void {
            for (uops) |uop| {
                switch (uop.op) {
                    .ADD, .SUB, .MUL, .FDIV, .POW => try ArithmeticRender.render(self.allocator, self.writer, uop),
                    .REDUCE_ADD, .REDUCE_MAX => try ReduceRender.render(self.allocator, self.writer, uop),
                    .MAX, .MIN => try ConditionalRender.render(self.allocator, self.writer, uop),
                    .EXP2, .NEG => try UnaryRender.render(self.allocator, self.writer, uop),
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

    var uops = [_]UOp{
        .{ .id = 0, .op = .ADD, .dtype = .f32, .src = &.{ 1, 2 }, .arg = null },
        .{ .id = 1, .op = .SUB, .dtype = .f32, .src = &.{ 3, 4 }, .arg = null },
        .{ .id = 2, .op = .MUL, .dtype = .f32, .src = &.{ 5, 6 }, .arg = null },
        .{ .id = 3, .op = .FDIV, .dtype = .f32, .src = &.{ 7, 8 }, .arg = null },
        .{ .id = 4, .op = .POW, .dtype = .f32, .src = &.{ 9, 10 }, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const t0 = @as(f32, t1) + @as(f32, t2);
        \\const t1 = @as(f32, t3) - @as(f32, t4);
        \\const t2 = @as(f32, t5) * @as(f32, t6);
        \\const t3 = @as(f32, t7) / @as(f32, t8);
        \\const t4 = std.math.pow(f32, t9, t10);
        \\
    ;
    const actual = try buffer.toOwnedSlice();
    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Reduce operations" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .REDUCE_ADD, .dtype = .f32, .src = &.{0}, .arg = null },
        .{ .id = 1, .op = .REDUCE_MAX, .dtype = .f32, .src = &.{1}, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const vec0: @Vector(t0.len, f32) = t0;
        \\const t0: f32 = @reduce(.Add, vec0);
        \\const vec1: @Vector(t1.len, f32) = t1;
        \\const t1: f32 = @reduce(.Max, vec1);
        \\
    ;
    const actual = try buffer.toOwnedSlice();

    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Conditional operations" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .MAX, .dtype = .f32, .src = &.{ 1, 2 }, .arg = null },
        .{ .id = 1, .op = .MIN, .dtype = .f32, .src = &.{ 3, 4 }, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const t0 = if(@as(f32,t1) > @as(f32,t2)) @as(f32,t1) else @as(f32,t2);
        \\const t1 = if(@as(f32,t3) < @as(f32,t4)) @as(f32,t3) else @as(f32,t4);
        \\
    ;
    const actual = try buffer.toOwnedSlice();

    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "Unary Operations" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit();

    var uops = [_]UOp{
        .{ .id = 0, .op = .EXP2, .dtype = .f32, .src = &.{1}, .arg = null },
        .{ .id = 1, .op = .NEG, .dtype = .f32, .src = &.{2}, .arg = null },
    };

    try renderer.render(&uops);

    const expected =
        \\const t0 = @as(f32, t1) * @as(f32, t1);
        \\const t1 = @as(f32,-t2);
        \\
    ;
    const actual = try buffer.toOwnedSlice();

    defer allocator.free(actual);

    try std.testing.expectEqualSlices(u8, expected, actual);
}

