const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;
const ViewInfo = @import("view_manager.zig").ViewInfo;

pub fn render(allocator: std.mem.Allocator, writer: anytype, uop: UOp, view_map: std.AutoHashMap(usize, ViewInfo)) !void {
    if (uop.op != .GEP) {
        return error.InvalidOperation;
    }
    if (uop.arg == null) {
        return error.NoAnyProvided;
    }

    const mem_info = uop.arg.?.mem_info;
    const base_ptr = mem_info.base;
    const offset = mem_info.offset;
    const stride = mem_info.stride;

    const result_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.id});
    defer allocator.free(result_var);
    const type_str = DTypeInfo.asString(uop.dtype);

    // Check if we're working with a view
    const view_id = uop.src[0];
    if (view_map.get(view_id)) |view_info| {
        if (uop.src.len - 1 != view_info.arg.view_meta.strides.len) {
            return error.InvalidOperation;
        }

        const strides = view_info.arg.view_meta.strides;
        const indexes = uop.src;
        var element_index: usize = 0;
        var i: usize = 1;
        while (i < indexes.len) : (i += 1) {
            const current_index = indexes[i];
            const current_stride = strides[i - 1];
            element_index += current_index * @as(usize, @intCast(current_stride));
        }

        const container_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
        defer allocator.free(container_var);

        try writer.print("const {s} = {d} + ({d} + {s}[{d}] * {d}) * @as(usize, @sizeOf({s}));\n", .{
            result_var,
            base_ptr,
            offset,
            container_var,
            element_index,
            stride,
            type_str,
        });
    } else {
        const container_var = try std.fmt.allocPrint(allocator, "t{d}", .{uop.src[0]});
        defer allocator.free(container_var);

        // Standard GEP operation without view
        try writer.print("const {s} = {d} + ({d} + {s}[{d}] * {d}) * @as(usize, @sizeOf({s}));\n", .{
            result_var,
            base_ptr,
            offset,
            container_var,
            uop.src[1],
            stride,
            type_str,
        });
    }
}
