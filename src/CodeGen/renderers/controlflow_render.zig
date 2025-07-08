const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(
    allocator: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    // Discard allocator if not needed
    _ = allocator;
    const ops = [_]UOpType{ .RANGE, .ENDRANGE };
    // Check if operation is supported
    if (!std.mem.containsAtLeast(UOpType, &ops, 1, &[_]UOpType{uop.op})) {
        return error.InvalidOperation;
    }

    if ((uop.op == .RANGE and uop.src.len != 0) or
        (uop.op == .ENDRANGE and uop.src.len != 1))
    {
        return error.InvalidOperandCount;
    }

    switch (uop.op) {
        .RANGE => {
            const index_name = ptr_map.get(uop.id) orelse return error.VariableNotFound;
            const start = uop.arg.?.loop_bounds.start;
            const end = uop.arg.?.loop_bounds.end;

            try writer.print("var {s}: {s} = {d}; // RANGE (uop {d})\nwhile ({s} < {d}) : ({s} += 1) {{\n", .{ index_name, DTypeInfo.asString(uop.dtype), start, uop.id, index_name, end, index_name });
        },
        .ENDRANGE => try writer.print("}}\n", .{}),
        else => unreachable,
    }
}
