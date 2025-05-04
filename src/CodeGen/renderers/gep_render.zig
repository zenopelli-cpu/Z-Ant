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
};

/// cast a loop index variable (i32) to a usize expression string
fn castIndex(
    alloc: std.mem.Allocator,
    ptr_map: *std.AutoHashMap(usize, []const u8),
    id: usize,
) ![]const u8 {
    const name = ptr_map.get(id) orelse return RendererError.VarMissing;
    return std.fmt.allocPrint(alloc, "@as(usize,@intCast({s}))", .{name});
}

/// emit "expr * stride" (skip if stride == 0)
fn emitTerm(w: anytype, expr: []const u8, stride: usize, first: *bool) !void {
    if (stride == 0) return;
    if (!first.*) try w.print(" + ", .{});
    try w.print("({s}*{d})", .{ expr, stride });
    first.* = false;
}

/// --------  MAIN entry  ---------------------------------------------------
pub fn render(
    alloc: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    view_map: *std.AutoHashMap(usize, ViewInfo),
    buffer_map: *std.AutoHashMap(usize, BufferInfo),
    ptr_map: *std.AutoHashMap(usize, []const u8),
) !void {
    // only GEPs handled here
    if (uop.op != .GEP) return RendererError.InvalidOp;
    if (uop.arg == null) return RendererError.NoAny;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const aalloc = arena.allocator();
    // ------------------------------------------------------------------ //
    // 1. base pointer name                                               //
    // ------------------------------------------------------------------ //

    // 1) Determine base pointer name (fix for slice vs pointer)
    const base_id = uop.src[0];
    var base_ptr: []const u8 = undefined;

    if (buffer_map.get(base_id)) |bi| {
        // bi.name is a slice (e.g. input_0), so for inputs/outputs append .ptr
        if (bi.is_input or std.mem.startsWith(u8, bi.name, "output_")) {
            // allocate in the existing arena
            base_ptr = try std.fmt.allocPrint(aalloc, "{s}.ptr", .{bi.name});
        } else {
            // internal buffers (addr_, acc_) are already pointers
            base_ptr = bi.name;
        }
    } else if (ptr_map.get(base_id)) |p| {
        // temporaries from ptr_map (like addr_3) are already raw pointers
        base_ptr = p;
    } else {
        return RendererError.VarMissing;
    }

    // ------------------------------------------------------------------ //
    // 2. build element-offset expression                                 //
    // ------------------------------------------------------------------ //

    var list = std.ArrayList(u8).init(aalloc);
    const w = list.writer();

    if (view_map.get(base_id)) |vinfo| {
        const shape = vinfo.arg.view_meta.shape;
        const strides = vinfo.arg.view_meta.strides;

        // (a) 1-D linear index into the view  --------------------------
        if (uop.src.len == 2) {
            if (shape.len != strides.len) return RendererError.RankMismatch;
            if (shape.len == 0) return RendererError.RankMismatch;

            const idx_expr = try castIndex(aalloc, ptr_map, uop.src[1]);

            // special-case rank-1 and rank-2 (enough for the tests)
            if (shape.len == 1) {
                // contiguous or broadcast vector
                var first_term = true;
                try emitTerm(w, idx_expr, @as(usize, @intCast(strides[0])), &first_term);
            } else if (shape.len == 2) {
                const cols = shape[1];
                const s_row = strides[0];
                const s_col = strides[1];
                // (i/cols)*s_row + (i%cols)*s_col
                try w.print(
                    "((({s}/{d})*{d})+(({s}%{d})*{d}))",
                    .{ idx_expr, cols, s_row, idx_expr, cols, s_col },
                );
            } else {
                // fall back to slow loop-unflatten (not needed yet)
                return RendererError.RankMismatch;
            }
        }
        // (b) fully-specified per-dim indexes --------------------------
        else if (uop.src.len - 1 == strides.len) {
            var first = true;
            for (uop.src[1..], 0..) |id, ax| {
                const idx_expr = try castIndex(aalloc, ptr_map, id);
                try emitTerm(w, idx_expr, @as(usize, @intCast(strides[ax])), &first);
            }
            if (first) try w.print("0", .{});
        } else {
            return RendererError.RankMismatch;
        }
    } else {
        // ----------------------------------------------------------------
        // raw buffer (no VIEW) – support 1-D or 2-D row-major
        // ----------------------------------------------------------------
        const nd = uop.src.len - 1;
        if (nd == 1) {
            const e = try castIndex(aalloc, ptr_map, uop.src[1]);
            try w.print("{s}", .{e});
        } else if (nd == 2) {
            const info = buffer_map.get(base_id) orelse return RendererError.VarMissing;
            if (info.shape.len < 2) return RendererError.RankMismatch;
            const cols = info.shape[info.shape.len - 1];
            const r_exp = try castIndex(aalloc, ptr_map, uop.src[1]);
            const c_exp = try castIndex(aalloc, ptr_map, uop.src[2]);
            try w.print("(({s}*{d})+{s})", .{ r_exp, cols, c_exp });
        } else {
            return RendererError.RankMismatch;
        }
    }

    const offset_expr = try list.toOwnedSlice();
    const dest_var = ptr_map.get(uop.id) orelse return RendererError.VarMissing;
    const dtype_name = DTypeInfo.asString(uop.dtype);

    // ------------------------------------------------------------------ //
    // 3. final pointer arithmetic                                        //
    // ------------------------------------------------------------------ //
    try writer.print(
        "    const {s} = @intFromPtr({s}) + ({s})*@sizeOf({s}); // GEP id={d}\n",
        .{ dest_var, base_ptr, offset_expr, dtype_name, uop.id },
    );
}
