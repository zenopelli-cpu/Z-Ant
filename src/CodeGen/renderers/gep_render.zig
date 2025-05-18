const std = @import("std");
const zant = @import("zant");

const UOp = zant.uops.UOp;
const ViewInfo = @import("view_manager.zig").ViewInfo;
const BufferInfo = @import("zig_renderer.zig").BufferInfo;
const DTypeInfo = zant.uops.DTypeInfo;

pub const RendererError = error{
    InvalidOp,
    NoAny,
    VarMissing,
    RankMismatch,
    OutOfMemory,
};

/// cast a loop index variable (i32) to a usize expression string
fn castIndex(
    alloc: std.mem.Allocator,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
    id: usize,
) ![]const u8 {
    const name = ptr_map.get(id) orelse return RendererError.VarMissing;
    return std.fmt.allocPrint(alloc, "@as(usize,@intCast({s}))", .{name});
}

/// emit "expr * stride" (skip if stride == 0)
fn emitTerm(w: anytype, expr: []const u8, stride: isize, first: *bool) !void {
    if (stride == 0) return;
    if (!first.*) try w.print(" + ", .{});
    try w.print("({s}*{d})", .{ expr, stride });
    first.* = false;
}

/// Builds an expression for 1D linear index into a view based on shape and strides
fn buildLinearIndexExpr(
    w: anytype,
    temp_alloc: std.mem.Allocator,
    linear_idx_expr: []const u8,
    view_shape: []const usize,
    view_strides: []const isize,
) !void {
    if (view_shape.len == 0) {
        try w.print("0", .{});
        return;
    }

    if (view_shape.len != view_strides.len) return RendererError.RankMismatch;

    try w.print("(", .{});
    var first_term = true;

    var current_divisor: []const u8 = "1";
    var i = view_shape.len - 1;
    while (true) {
        const dim_size = view_shape[i];
        const stride = view_strides[i];

        if (stride != 0) {
            var dim_index_expr_buf: [256]u8 = undefined;
            var dim_index_expr: []const u8 = undefined;

            if (std.mem.eql(u8, current_divisor, "1")) {
                if (dim_size == 1) {
                    dim_index_expr = "0";
                } else {
                    dim_index_expr = try std.fmt.bufPrint(&dim_index_expr_buf, "({s} % {d})", .{ linear_idx_expr, dim_size });
                }
            } else {
                if (dim_size == 1) {
                    dim_index_expr = "0";
                } else {
                    dim_index_expr = try std.fmt.bufPrint(&dim_index_expr_buf, "(({s} / {s}) % {d})", .{ linear_idx_expr, current_divisor, dim_size });
                }
            }

            try emitTerm(w, dim_index_expr, stride, &first_term);
        }

        if (i == 0) break;
        i -= 1;

        if (dim_size > 1) {
            var next_divisor: []const u8 = undefined;
            if (std.mem.eql(u8, current_divisor, "1")) {
                next_divisor = try std.fmt.allocPrint(temp_alloc, "{d}", .{dim_size});
            } else {
                next_divisor = try std.fmt.allocPrint(temp_alloc, "({s} * {d})", .{ current_divisor, dim_size });
            }
            current_divisor = next_divisor;
        }
    }

    if (first_term) try w.print("0", .{});
    try w.print(")", .{});
}

/// Builds an offset expression for multidimensional index
fn buildMultiDimOffsetExpr(
    w: anytype,
    alloc: std.mem.Allocator,
    ptr_map: *const std.AutoHashMap(usize, []const u8),
    gep_indices: []const usize,
    target_strides: []const isize,
) !void {
    var first = true;
    if (gep_indices.len - 1 != target_strides.len) {
        try w.print("0", .{});
        return;
    }
    for (gep_indices[1..], 0..) |idx_uop_id, axis_idx| {
        const idx_expr_str = try castIndex(alloc, ptr_map, idx_uop_id);
        defer alloc.free(idx_expr_str);
        try emitTerm(w, idx_expr_str, target_strides[axis_idx], &first);
    }
    if (first) try w.print("0", .{});
}

/// Gets base variable name for .ptr access, like "input_0" or "output_10"
fn getBaseVariableNameForPtrAccess(
    alloc: std.mem.Allocator,
    ultimate_base_id: usize,
    buffer_map: *const std.AutoHashMap(usize, BufferInfo),
) ![]const u8 {
    if (buffer_map.get(ultimate_base_id)) |bi| {
        if (bi.is_input) {
            return std.fmt.allocPrint(alloc, "input_{d}", .{ultimate_base_id});
        } else {
            return std.fmt.allocPrint(alloc, "{s}", .{bi.name});
        }
    }
    std.debug.print("getBaseVariableNameForPtrAccess: Error - ultimate_base_id {d} not found in buffer_map\n", .{ultimate_base_id});
    return RendererError.VarMissing;
}

/// Main entry point for rendering GEP operations
pub fn render(
    alloc: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    view_map: *std.AutoHashMap(usize, ViewInfo),
    buffer_map: *const std.AutoHashMap(usize, BufferInfo),
    ptr_map: *const std.AutoHashMap(usize, []const u8),
) !void {
    if (uop.op != .GEP) return RendererError.InvalidOp;

    if (uop.arg) |payload| {
        switch (payload) {
            .mem_info_gep_info => {
                const mem_info = payload.mem_info_gep_info;

                const base_var_name = try getBaseVariableNameForPtrAccess(alloc, mem_info.base, buffer_map);
                defer alloc.free(base_var_name);
                const gep_result_name = ptr_map.get(uop.id) orelse return RendererError.VarMissing;
                const dtype_name = DTypeInfo.asString(uop.dtype);

                try writer.print(
                    "    const {s} = @intFromPtr({s}.ptr) + (@as(usize, @intCast(idx_{d}))) * @sizeOf({s}); // GEP id={d}\n",
                    .{ gep_result_name, base_var_name, uop.src[1], dtype_name, uop.id },
                );
                return;
            },
            else => {},
        }
    }

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp_alloc = arena.allocator();

    const gep_result_name = ptr_map.get(uop.id) orelse return RendererError.VarMissing;
    const view_or_buffer_id_as_base = uop.src[0];

    var base_var_name_for_final_gep: []const u8 = undefined;
    var offset_calculation_list = std.ArrayList(u8).init(temp_alloc);
    const offset_writer = offset_calculation_list.writer();

    if (view_map.get(view_or_buffer_id_as_base)) |vinfo| {
        const ultimate_source_id = vinfo.src[0];
        base_var_name_for_final_gep = try getBaseVariableNameForPtrAccess(temp_alloc, ultimate_source_id, buffer_map);

        const view_strides = vinfo.arg.view_meta.strides;
        const view_shape = vinfo.arg.view_meta.shape;

        if (view_shape.len != view_strides.len or view_shape.len == 0) {
            std.debug.print("GEP RankMismatch (View Base): View {d} shape/strides rank mismatch or zero rank.\n", .{view_or_buffer_id_as_base});
            return RendererError.RankMismatch;
        }

        if (uop.src.len - 1 == 0 and view_shape.len > 0) {
            try offset_writer.print("0", .{});
        } else if (uop.src.len == 2) {
            const linear_idx_uop_id = uop.src[1];
            const linear_idx_expr = try castIndex(temp_alloc, ptr_map, linear_idx_uop_id);
            defer temp_alloc.free(linear_idx_expr);
            try buildLinearIndexExpr(offset_writer, temp_alloc, linear_idx_expr, view_shape, view_strides);
        } else if (uop.src.len - 1 == view_strides.len) {
            try buildMultiDimOffsetExpr(offset_writer, temp_alloc, ptr_map, uop.src, view_strides);
        } else {
            std.debug.print("GEP RankMismatch (View Base): Got {d} indices for view {d} with strides rank {d}\n", .{ uop.src.len - 1, view_or_buffer_id_as_base, view_strides.len });
            return RendererError.RankMismatch;
        }
    } else if (buffer_map.get(view_or_buffer_id_as_base)) |buf_info| {
        base_var_name_for_final_gep = try getBaseVariableNameForPtrAccess(temp_alloc, view_or_buffer_id_as_base, buffer_map);

        const buffer_shape = buf_info.shape;
        if (uop.src.len - 1 == 0) {
            try offset_writer.print("0", .{});
        } else if (uop.src.len == 2 and buffer_shape.len > 0) {
            const linear_idx_uop_id = uop.src[1];
            const linear_idx_expr = try castIndex(temp_alloc, ptr_map, linear_idx_uop_id);
            defer temp_alloc.free(linear_idx_expr);
            try offset_writer.print("{s}", .{linear_idx_expr});
        } else if (uop.src.len - 1 == buffer_shape.len and buffer_shape.len > 0) {
            var calculated_strides = try temp_alloc.alloc(isize, buffer_shape.len);
            calculated_strides[buffer_shape.len - 1] = 1;
            var i = buffer_shape.len - 1;
            while (i > 0) : (i -= 1) {
                calculated_strides[i - 1] = calculated_strides[i] * @as(isize, @intCast(buffer_shape[i]));
            }
            try buildMultiDimOffsetExpr(offset_writer, temp_alloc, ptr_map, uop.src, calculated_strides);
        } else if (buffer_shape.len == 0 and uop.src.len > 1) {
            std.debug.print("GEP Error: Attempt to index scalar buffer {d} with {d} indices\n", .{ view_or_buffer_id_as_base, uop.src.len - 1 });
            return RendererError.RankMismatch;
        } else {
            std.debug.print("GEP RankMismatch (Buffer Base): Got {d} indices for buffer {d} with rank {d}\n", .{ uop.src.len - 1, view_or_buffer_id_as_base, buffer_shape.len });
            return RendererError.RankMismatch;
        }
    } else {
        std.debug.print("GEP Error: Base ID {d} not found in view_map or buffer_map\n", .{view_or_buffer_id_as_base});
        return RendererError.VarMissing;
    }

    const offset_expr_slice = try offset_calculation_list.toOwnedSlice();
    const dtype_name = DTypeInfo.asString(uop.dtype);

    try writer.print(
        "    const {s} = @intFromPtr({s}.ptr) + ({s})*@sizeOf({s}); // GEP id={d}\n",
        .{ gep_result_name, base_var_name_for_final_gep, offset_expr_slice, dtype_name, uop.id },
    );
}
