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

/// Builds an expression for 1D linear index into a view based on shape and strides
fn buildLinearIndexExpr(
    w: anytype,
    idx_expr: []const u8,
    shape: []const usize,
    strides: []const isize,
) !void {
    if (shape.len == 1) {
        // Contiguous or broadcast vector
        var first_term = true;
        try emitTerm(w, idx_expr, @as(usize, @intCast(strides[0])), &first_term);
        if (first_term) try w.print("0", .{});
    } else if (shape.len == 2) {
        const cols = shape[1];
        const s_row = strides[0];
        const s_col = strides[1];
        try w.print(
            "((({s}/{d})*{d})+(({s}%{d})*{d}))",
            .{ idx_expr, cols, s_row, idx_expr, cols, s_col },
        );
    } else {
        return RendererError.RankMismatch;
    }
}

/// Builds an offset expression for multidimensional index
fn buildMultiDimExpr(
    w: anytype,
    alloc: std.mem.Allocator,
    ptr_map: *std.AutoHashMap(usize, []const u8),
    sources: []const usize,
    strides: []const isize,
) !void {
    var first = true;
    for (sources[1..], 0..) |id, ax| {
        const idx_expr = try castIndex(alloc, ptr_map, id);
        try emitTerm(w, idx_expr, @as(usize, @intCast(strides[ax])), &first);
    }
    if (first) try w.print("0", .{});
}

/// Gets base pointer name, handling slices vs raw pointers
fn getBasePointer(
    alloc: std.mem.Allocator,
    base_id: usize,
    buffer_map: *std.AutoHashMap(usize, BufferInfo),
    ptr_map: *std.AutoHashMap(usize, []const u8),
) ![]const u8 {
    if (buffer_map.get(base_id)) |bi| {
        // For slices (inputs/outputs), append .ptr
        if (bi.is_input or std.mem.startsWith(u8, bi.name, "output_")) {
            return std.fmt.allocPrint(alloc, "{s}.ptr", .{bi.name});
        }
        return bi.name; // Internal buffers are already pointers
    } else if (ptr_map.get(base_id)) |p| {
        return p; // Raw pointers from ptr_map
    }
    return RendererError.VarMissing;
}

/// Main entry point for rendering GEP operations
pub fn render(
    alloc: std.mem.Allocator,
    writer: anytype,
    uop: UOp,
    view_map: *std.AutoHashMap(usize, ViewInfo),
    buffer_map: *std.AutoHashMap(usize, BufferInfo),
    ptr_map: *std.AutoHashMap(usize, []const u8),
) !void {
    // Validate operation type
    if (uop.op != .GEP) return RendererError.InvalidOp;
    if (uop.arg == null) return RendererError.NoAny;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const temp_alloc = arena.allocator();

    // Get base pointer ID
    const base_id = uop.src[0];

    // Build element-offset expression string into 'list'
    var list = std.ArrayList(u8).init(temp_alloc);
    const w = list.writer();

    // --- MODIFIED LOGIC: Handle VIEW case first ---
    if (view_map.get(base_id)) |vinfo| {
        // CASE 1: Base *is* a VIEW ID
        const shape = vinfo.arg.view_meta.shape;
        const strides = vinfo.arg.view_meta.strides;
        const actual_base_id = vinfo.src[0]; // Get the original source ID from VIEW info
        const base_ptr = try getBasePointer(temp_alloc, actual_base_id, buffer_map, ptr_map);

        if (shape.len != strides.len or shape.len == 0)
            return RendererError.RankMismatch;

        // Handle 1-D linear indexing into the VIEW
        if (uop.src.len == 2) {
            const idx_expr = try castIndex(temp_alloc, ptr_map, uop.src[1]);
            try buildLinearIndexExpr(w, idx_expr, shape, strides);
        }
        // Handle fully-specified per-dim indexing into the VIEW
        else if (uop.src.len - 1 == strides.len) {
            // Pass VIEW strides directly
            try buildMultiDimExpr(w, temp_alloc, ptr_map, uop.src, strides);
        } else {
            std.debug.print("GEP RankMismatch (View Base): Got {d} indices for view {d} with strides rank {d}\n", .{ uop.src.len - 1, base_id, strides.len });
            return RendererError.RankMismatch;
        }
        // Emit the final pointer arithmetic using the resolved base pointer and calculated offset
        const offset_expr = try list.toOwnedSlice();
        const dest_var = ptr_map.get(uop.id) orelse return RendererError.VarMissing;
        const dtype_name = DTypeInfo.asString(uop.dtype);
        try writer.print(
            "    const {s} = @intFromPtr({s}) + ({s})*@sizeOf({s}); // GEP id={d} (via VIEW)\n",
            .{ dest_var, base_ptr, offset_expr, dtype_name, uop.id },
        );
    } else if (buffer_map.get(base_id)) |buf_info| {
        // CASE 2: Base is a Buffer ID (DEFINE_GLOBAL, etc.)
        const shape = buf_info.shape;
        const num_indices = uop.src.len - 1;
        const base_ptr = try getBasePointer(temp_alloc, base_id, buffer_map, ptr_map); // Use direct base_id

        // Handle linear or N-dim indexing based on buffer shape
        if (num_indices == 1) {
            // Linear indexing
            const idx_expr = try castIndex(temp_alloc, ptr_map, uop.src[1]);
            try w.print("{s}", .{idx_expr});
        } else if (num_indices == shape.len) {
            // N-dimensional Indexing
            // ... (existing logic to calculate strides from shape and build offset) ...
            if (num_indices == 0) {
                try w.print("0", .{});
            } else {
                var strides = try temp_alloc.alloc(usize, shape.len);
                if (strides.len != shape.len) return std.mem.Allocator.Error.OutOfMemory;
                strides[shape.len - 1] = 1;
                var i = shape.len - 1;
                while (i > 0) : (i -= 1) {
                    strides[i - 1] = strides[i] * shape[i];
                }
                var first = true;
                for (uop.src[1..], 0..) |idx_id, ax| {
                    const idx_expr = try castIndex(temp_alloc, ptr_map, idx_id);
                    try emitTerm(w, idx_expr, strides[ax], &first);
                }
                if (first) try w.print("0", .{});
            }
        } else {
            // Error: Index mismatch for Buffer base
            std.debug.print("GEP RankMismatch (Buffer Base): Got {d} indices for buffer {d} with shape {any} (rank {d}), expected 1 or {d}\n", .{ num_indices, base_id, shape, shape.len, shape.len });
            return RendererError.RankMismatch;
        }
        // Emit the final pointer arithmetic using the direct base pointer and calculated offset
        const offset_expr = try list.toOwnedSlice();
        const dest_var = ptr_map.get(uop.id) orelse return RendererError.VarMissing;
        const dtype_name = DTypeInfo.asString(uop.dtype);
        try writer.print(
            "    const {s} = @intFromPtr({s}) + ({s})*@sizeOf({s}); // GEP id={d} (Buffer)\n",
            .{ dest_var, base_ptr, offset_expr, dtype_name, uop.id },
        );
    } else {
        // CASE 3: Base ID not found in view_map OR buffer_map
        std.debug.print("GEP Error: Base ID {d} not found in view_map or buffer_map\n", .{base_id});
        return RendererError.VarMissing;
    }
}
