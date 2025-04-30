const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const Renderer = codegen.renderer;
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;

const ZigRenderer = Renderer.ZigRenderer;

test "Arithmetic operations" {
    std.debug.print("Running zig renderer arithmetic test \n", .{});
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
    std.debug.print("Running zig renderer reduce test \n", .{});
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
    std.debug.print("Running zig renderer conditional test \n", .{});
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
    std.debug.print("Running zig renderer unary test \n", .{});
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
