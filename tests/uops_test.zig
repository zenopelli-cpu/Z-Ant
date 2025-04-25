const std = @import("std");
const testing = std.testing;
const Uops = @import("zant").Uops;
const zant = @import("zant");
const UOp = Uops.UOp;
const UOpType = Uops.UOpType;
const DType = Uops.DType;
const Any = Uops.Any;
const HLOp = Uops.HLOp;
const HLType = Uops.HLType;
const lowerHL = Uops.lowerHL;
var allocator = zant.utils.allocator.allocator;

test "create simple UOp" {
    const op = UOp{
        .op = .ADD,
        .dtype = .f32,
        .src = &.{ 1, 2 },
    };
    try testing.expectEqual(op.op, .ADD);
    try testing.expectEqual(op.dtype, .f32);
    try testing.expectEqualSlices(usize, op.src, &.{ 1, 2 });
    try testing.expectEqual(op.arg, null);
}

test "create UOp with int arg" {
    const op = UOp{
        .op = .CONST,
        .dtype = .i32,
        .src = &.{},
        .arg = .{ .int = 42 },
    };
    try testing.expectEqual(op.op, .CONST);
    try testing.expectEqual(op.dtype, .i32);
    try testing.expectEqual(op.src.len, 0);
    try testing.expect(op.arg.? == .int);
    try testing.expectEqual(op.arg.?.int, 42);
}

test "create UOp with float arg" {
    const op = UOp{
        .op = .CONST,
        .dtype = .f32,
        .src = &.{},
        .arg = .{ .float = 3.14 },
    };
    try testing.expectEqual(op.op, .CONST);
    try testing.expectEqual(op.dtype, .f32);
    try testing.expect(op.arg.? == .float);
    try testing.expectEqual(op.arg.?.float, 3.14);
}

test "create UOp with label arg" {
    const op = UOp{
        .op = .DEFINE_GLOBAL,
        .dtype = .i8,
        .src = &.{},
        .arg = .{ .label = "my_tensor" },
    };
    try testing.expectEqual(op.op, .DEFINE_GLOBAL);
    try testing.expectEqual(op.dtype, .i8);
    try testing.expect(op.arg.? == .label);
    try testing.expectEqualStrings(op.arg.?.label, "my_tensor");
}

test "create UOp with shape arg" {
    const shape_data = [_]usize{ 10, 20 };
    const op = UOp{
        .op = .DEFINE_GLOBAL,
        .dtype = .f32,
        .src = &.{},
        .arg = .{ .shape = &shape_data },
    };
    try testing.expectEqual(op.op, .DEFINE_GLOBAL);
    try testing.expectEqual(op.dtype, .f32);
    try testing.expect(op.arg.? == .shape);
    try testing.expectEqualSlices(usize, op.arg.?.shape, &shape_data);
}

test "create UOp with loop bounds arg" {
    const op = UOp{
        .op = .RANGE,
        .dtype = .i32, // loop counter type?
        .src = &.{},
        .arg = .{ .loop_bounds = .{ .start = 0, .end = 10 } },
    };
    try testing.expectEqual(op.op, .RANGE);
    try testing.expect(op.arg.? == .loop_bounds);
    try testing.expectEqual(op.arg.?.loop_bounds.start, 0);
    try testing.expectEqual(op.arg.?.loop_bounds.end, 10);
}

// TODO: Add tests for fused_ops, mem_info, tile args
// TODO: Add test for UOp.format

// Example of testing format (requires Allocator)
// test "UOp format" {
//     var arena = std.heap.ArenaAllocator.init(testing.allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();
//
//     const op = UOp{
//         .op = .ADD,
//         .dtype = .f32,
//         .src = &.{ 1, 2 },
//         .arg = .{ .int = 5 }, // Example arg
//     };
//
//     const formatted_str = try std.fmt.allocPrint(allocator, "{d}", .{op});
//     defer allocator.free(formatted_str);
//
//     try testing.expectEqualStrings(formatted_str, "UOpType.ADD: ADD  arg=Uops.Any{ .int = 5 }");
// }

test "HL lowering" {
    const N = 32;
    const hl = [_]HLOp{
        .{ .typ = .Const, .val = 1.5, .elemN = N, .lhs = null, .rhs = null }, // id 0
        .{ .typ = .Const, .val = -0.2, .elemN = N, .lhs = null, .rhs = null }, // id 1
        .{ .typ = .Add, .lhs = 0, .rhs = 1, .elemN = N }, // id 2
        .{ .typ = .Relu, .lhs = 2, .rhs = null, .elemN = N }, // id 3
    };

    const uops = try lowerHL(allocator, &hl, N);
    defer {
        for (uops) |uop| {
            if (uop.src.len > 0) {
                allocator.free(uop.src);
            }
        }
        allocator.free(uops);
    }

    // Verify basic properties
    try testing.expect(uops.len > 0);

    // Verify first op is DEFINE_GLOBAL
    try testing.expectEqual(uops[0].op, .DEFINE_GLOBAL);
    try testing.expectEqual(uops[0].dtype, .f32);

    // Verify last op is ENDRANGE
    try testing.expectEqual(uops[uops.len - 1].op, .ENDRANGE);

    // Count specific ops to verify structure
    var def_globals: usize = 0;
    var ranges: usize = 0;
    var endranges: usize = 0;
    var loads: usize = 0;
    var stores: usize = 0;
    var adds: usize = 0;
    var cmplts: usize = 0;
    var wheres: usize = 0;

    for (uops) |u| {
        switch (u.op) {
            .DEFINE_GLOBAL => def_globals += 1,
            .RANGE => ranges += 1,
            .ENDRANGE => endranges += 1,
            .LOAD => loads += 1,
            .STORE => stores += 1,
            .ADD => adds += 1,
            .CMPLT => cmplts += 1,
            .WHERE => wheres += 1,
            else => {},
        }
    }

    // Verify we have the expected number of each op type
    try testing.expectEqual(def_globals, 4); // One for each HL op
    try testing.expectEqual(ranges, 2); // One for Add, one for Relu
    try testing.expectEqual(endranges, 2);
    try testing.expectEqual(loads, 3); // Two for Add, one for Relu
    try testing.expectEqual(stores, 2); // One for Add, one for Relu
    try testing.expectEqual(adds, 1);
    try testing.expectEqual(cmplts, 1);
    try testing.expectEqual(wheres, 1);

    // Pretty print the generated uops
    std.debug.print("\nGenerated UOps (N = {}):\n", .{N});
    for (uops, 0..) |u, i| {
        std.debug.print("{d}: {}\n", .{ i, u });
    }
}

test "ZigRenderer test" {
    // 1. Define a simple HL program (e.g., add two constants then apply Relu)
    const N = 1; // Keep it simple for string comparison
    const hl = [_]HLOp{
        .{ .typ = .Const, .val = 5.0, .elemN = N, .lhs = null, .rhs = null }, // HL id 0 -> buf_id 0, const_id 4
        .{ .typ = .Const, .val = -10.0, .elemN = N, .lhs = null, .rhs = null }, // HL id 1 -> buf_id 1, const_id 5 (Changed val to negative)
        .{ .typ = .Add, .lhs = 0, .rhs = 1, .elemN = N }, // HL id 2 -> buf_id 2 (Add result)
        .{ .typ = .Relu, .lhs = 2, .rhs = null, .elemN = N }, // HL id 3 -> buf_id 3 (Relu output)
    };

    // 2. Lower the HL program to UOps
    const uops = try lowerHL(allocator, &hl, N);
    defer {
        for (uops) |uop| {
            if (uop.src.len > 0) {
                allocator.free(uop.src);
            }
        }
        allocator.free(uops);
    }

    // 3. Render the UOps to Zig code
    var sb = std.ArrayList(u8).init(allocator);
    defer sb.deinit();
    const writer = sb.writer(); // Get the specific writer type
    // Initialize ZigRenderer with the specific writer type
    const RendererType = Uops.ZigRenderer.new(@TypeOf(writer)); // Get the specific type via .new
    var renderer = RendererType.init(&allocator, writer); // Initialize using the specific type (pass allocator address)
    // Defer renderer deinit *after* rendering but *before* freeing the sb
    defer renderer.deinit();

    try renderer.render(uops);
    const generated_code = try sb.toOwnedSlice();
    defer allocator.free(generated_code);

    std.debug.print("\nGenerated Kernel:\n{s}\n", .{generated_code});
}
