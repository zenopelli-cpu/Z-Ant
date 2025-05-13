const std = @import("std");
const zant = @import("zant");

const ViewInfo = @import("view_manager.zig").ViewInfo;
const BufferInfo = @import("zig_renderer.zig").BufferInfo;

const UOp = zant.uops.UOp;
const DTypeInfo = zant.uops.DTypeInfo;

pub const RendererError = error{
    InvalidOp,
    NoAny,
    VarMissing,
    RankMismatch,
    OutOfMemory,
};

/// Main entry point for rendering GEP operations
pub fn manage(
    alloc: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    view_map: *std.AutoHashMap(usize, ViewInfo),
    _: *const std.AutoHashMap(usize, BufferInfo),
    _: *const std.AutoHashMap(usize, []const u8),
) !void {
    if (uop.op != .RESHAPE) return RendererError.InvalidOp;
    if (uop.src.len != 1) return RendererError.InvalidOp;

    if (uop.arg == null) return RendererError.NoAny;
    const shape = uop.arg.?.shape;
    if (shape.len == 0) return RendererError.RankMismatch;

    var strides = try alloc.alloc(isize, shape.len);
    errdefer alloc.free(strides);

    var stride: isize = 1;
    var i: usize = shape.len;

    while (i > 0) : (i -= 1) {
        strides[i - 1] = stride;
        stride *= @intCast(shape[i - 1]);
    }

    const src_id = uop.src[0];
    const get_or_put_result = try view_map.getOrPut(src_id);
    const src_view = get_or_put_result.value_ptr.*;

    // Create output view
    const out_view = ViewInfo{
        .dtype = src_view.dtype,
        .src = &.{src_id},
        .arg = .{
            .view_meta = .{
                .shape = shape,
                .strides = if (strides.len > 0) strides else &.{1},
            },
        },
    };

    // Store the new view in the view map
    try view_map.put(uop.id, out_view);

    // Write the reshape operation
    try writer.print(
        \\// RESHAPE op {d}
        \\// Source: {d}, Output shape: {any}
        \\// Note: Reshape is a view-only operation; the underlying data remains unchanged
        \\
    , .{ uop.id, src_id, shape });
}
