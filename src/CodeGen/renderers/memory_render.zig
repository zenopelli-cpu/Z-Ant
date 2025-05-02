const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp) !void {
    if (uop.op != .DEFINE_GLOBAL and uop.op != .LOAD and uop.op != .STORE) {
        return error.InvalidOperation;
    }

    if (uop.src.len != 1 and uop.op == .DEFINE_GLOBAL and uop.arg.?.label == null) {
        return error.InvalidOperandCount;
    }


    if (uop.src.len != 2 and uop.op == .STORE) {
        return error.InvalidOperandCount;
    }

    if (uop.src.len != 1 and uop.op == .LOAD) {
        return error.InvalidOperandCount;
    }

    switch(uop.op){
        .DEFINE_GLOBAL => {
            const type_str = DTypeInfo.asString(uop.dtype);
            const allocation_name = try std.fmt.allocPrint(allocator, "{s}", .{uop.arg.?.label});
            defer allocator.free(allocation_name);   

            const allocation_size = try std.fmt.allocPrint(allocator, "{d}", .{uop.arg.?.int});
            defer allocator.free(allocation_size); 

            try writer.print("const {s} = try allocator.alloc({s}, {s});\ndefer allocator.free({s});\n", .{allocation_name, type_str, allocation_size, allocation_name});
        },
        .STORE => {
            try writer.print("*t{d} = t{d}\n", .{uop.src[0], uop.src[1]});
        },
        .LOAD => {
            try writer.print("const t{d} = *t{d}\n", .{uop.id, uop.src[0]});
        },
        else => unreachable,
    }

}