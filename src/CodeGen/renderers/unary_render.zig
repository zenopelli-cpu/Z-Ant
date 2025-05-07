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

    const supported_ops = [_]UOpType{ .EXP2, .NEG, .CAST };

    // Validate operation type
    if (!std.mem.containsAtLeast(UOpType, &supported_ops, 1, &[_]UOpType{uop.op})) {
        return error.InvalidOperation;
    }

    //minimal Max and min are only attainable from two variables, not more or less.
    if (uop.src.len != 1) {
        return error.InvalidOperandCount;
    }

    const result_var = ptr_map.get(uop.id) orelse return error.VariableNotFound;
    const src_var = ptr_map.get(uop.src[0]) orelse return error.VariableNotFound;

    // const result_type_str = DTypeInfo.asString(uop.dtype);

    // const first_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
    // defer allocator.free(first_var);

    // Need source type for CAST
    // const src_type_str = DTypeInfo.asString(???); // How to get src uop's type?

    // Updated print statements to assign to existing buffer elements (assuming scalar ops for now)
    switch (uop.op) {
        .EXP2 => {
            const type_str = DTypeInfo.asString(uop.dtype);
            try writer.print("    const {s}: {s} = std.math.exp2({s}); // EXP2 (uop {d})\n", .{ result_var, type_str, src_var, uop.id });
            try writer.print("    _ = &{s};\n", .{result_var});
        },
        .NEG => {
            const type_str = DTypeInfo.asString(uop.dtype);
            try writer.print("    const {s}: {s} = -{s}; // NEG (uop {d})\n", .{ result_var, type_str, src_var, uop.id });
            try writer.print("    _ = &{s};\n", .{result_var});
        },
        .CAST => {
            @panic("CAST rendering needs source type information");
        },
        .CLIP => {
            if(uop.dtype!=uop.arg.?.clip_bounds.type) return error.TypeNotMatching;
            const type_str = DTypeInfo.asString(uop.dtype);
            const cast = try std.fmt.allocPrint(allocator, "@as({s}, {s})", .{type_str, src_var});
            defer allocator.free(cast);
            const max_str = try std.fmt.allocPrint(allocator, "{any}", .{uop.arg.?.clip_bounds.max});
            defer allocator.free(max_str);
            const min_str = try std.fmt.allocPrint(allocator, "{any}", .{uop.arg.?.clip_bounds.min});
            defer allocator.free(min_str);
            const min = try std.fmt.allocPrint(allocator, "@as({s}, {s})", .{type_str, max_str});
            defer allocator.free(min);
            const max = try std.fmt.allocPrint(allocator, "@as({s}, {s})", .{type_str, min_str});
            defer allocator.free(max);
            try writer.print(" const {s} = if( {s} < {s}) { if ({s} > {s}) {s} else {s}; } else {s};\n", 
                .{result_var, cast, min, cast, max, max, cast, min});
        },
        else => unreachable,
    }
}
