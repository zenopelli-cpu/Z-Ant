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
            const t: usize = 12123;
            const t8: u8 = t & 0xFF;
            _ = t8;
            if(uop.dtype!=uop.arg.?.clip_bounds.type){
                return error.TypeNotMatching;
            }

            const type_str = DTypeInfo.asString(uop.dtype);
            const is_accumulator_update = std.mem.startsWith(u8, src_var, "acc_");
            switch(DTypeInfo.byteSize(uop.dtype)){
                1 => {
                    const maxstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}&0xFF))", .{ type_str, uop.arg.?.clip_bounds.max });
                    defer allocator.free(maxstr);
                    const minstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}&0xFF))", .{ type_str, uop.arg.?.clip_bounds.min });
                    defer allocator.free(minstr);

                    if(is_accumulator_update){
                        try writer.print("  {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ src_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    } else {
                        try writer.print("  const {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ result_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    }
                },
                2 => {
                    const maxstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}&0xFFFF))", .{ type_str, uop.arg.?.clip_bounds.max });
                    defer allocator.free(maxstr);
                    const minstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}&0xFFFF))", .{ type_str, uop.arg.?.clip_bounds.min });
                    defer allocator.free(minstr);

                    if(is_accumulator_update){
                        try writer.print("  {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ src_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    } else {
                        try writer.print("  const {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ result_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    }
                },
                4 => {
                    const maxstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}&0xFFFFFFFF))", .{ type_str, uop.arg.?.clip_bounds.max });
                    defer allocator.free(maxstr);
                    const minstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}&0xFFFFFFFF))", .{ type_str, uop.arg.?.clip_bounds.min });
                    defer allocator.free(minstr);

                    if(is_accumulator_update){
                        try writer.print("  {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ src_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    } else {
                        try writer.print("  const {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ result_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    }
                },
                else => {
                    const maxstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}))", .{ type_str, uop.arg.?.clip_bounds.max });
                    defer allocator.free(maxstr);
                    const minstr = std.fmt.allocPrint(allocator, "@as({s}, @bitCast({d}))", .{ type_str, uop.arg.?.clip_bounds.min });
                    defer allocator.free(minstr);

                    if(is_accumulator_update){
                        try writer.print("  {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ src_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    } else {
                        try writer.print("  const {s} = if({s}<{s}){if({s}>{s}) {s} else {s};} else {s};\n", .{ result_var, src_var, maxstr, src_var, minstr, src_var, minstr, maxstr});
                    }
                },
            }
        },
        else => unreachable,
    }
}


// if(is_accumulator_update){
            //     try writer.print(" {s} = if({s} < {s}) { if ({s} > {s}) {s} else {s};} else {s}; // CLIP into Accumulator (uop {d})\n",
            //         .{result_var, src_var, min_str, src_var, max_str, max_str, src_var, min_str, uop.id});
            // } else {
            //     try writer.print(" const {s} = if({s} < {s}) { if ({s} > {s}) {s} else {s};} else {s};\n", 
            //         .{result_var, src_var, min_str, src_var, max_str, max_str, src_var, min_str});
            // }