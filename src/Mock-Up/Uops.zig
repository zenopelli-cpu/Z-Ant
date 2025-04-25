const std = @import("std");
const zant = @import("zant");
const Tensor = zant.tensor.Tensor;
const allocator = zant.utils.allocator.allocator;

/// 1) UOpType  – every primitive we agreed on (truncated for brevity)
///------------------------------------------------------------------------
pub const UOpType = enum { // truncated
    DEFINE_GLOBAL,
    CONST,
    LOAD,
    STORE,
    ADD,
    CMPLT,
    WHERE,
    RANGE,
    ENDRANGE,
    FUSE,
};

///------------------------------------------------------------------------
/// 2) Any – generic payload for `UOp.arg`
///------------------------------------------------------------------------
pub const Any = union(enum) {
    // ── simple scalars
    int: usize,
    float: f32,
    bool: bool,

    // ── byte labels / names
    label: []const u8,

    // ── 1-D or n-D shapes
    shape: []const usize,

    // ── list of fused ops            (used by FUSE marker)
    fused_ops: []const UOpType,

    // ── loop bounds for RANGE
    loop_bounds: struct { start: usize, end: usize },

    // ── parameters for LOAD/STORE/GEP
    mem_info: struct { base: []const u8, offset: usize, stride: usize },

    // ── tiling / vector width
    tile: struct { size: usize },

    // extend here when you need more specialised payloads
};

///------------------------------------------------------------------------
/// 3) DType  – minimalist numeric type enum
///------------------------------------------------------------------------
pub const DType = enum { f32, i32, i8, bool };

///------------------------------------------------------------------------
/// 4) UOp  – one linear instruction in the IR
///------------------------------------------------------------------------
pub const UOp = struct {
    op: UOpType, // micro-op tag
    dtype: DType, // data type operated on
    src: []const usize, // IDs of source UOps (for dependency or SSA; can be empty)
    arg: ?Any = null, // extra payload (loop bounds, constant value, etc.)

    /// helper pretty-printer
    pub fn format(self: UOp, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}", .{@tagName(self.op)});
        if (self.arg) |a| {
            switch (a) {
                .float => |f| try writer.print(" {d}", .{f}),
                .int => |n| try writer.print(" {d}", .{n}),
                .loop_bounds => |l| try writer.print(" [{d}..{d}]", .{ l.start, l.end }),
                else => try writer.print(" arg={any}", .{a}),
            }
        }
        if (self.src.len > 0) {
            try writer.print(" src=[", .{});
            for (self.src, 0..) |s, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{d}", .{s});
            }
            try writer.print("]", .{});
        }
    }
};

//////////////////////////////////////////////////////////////////////////
// 1. UOpBuilder
//////////////////////////////////////////////////////////////////////////
pub const UOpBuilder = struct {
    allocator: std.mem.Allocator,
    list: std.ArrayList(UOp),
    next_id: usize = 0,

    pub fn init(alloc: std.mem.Allocator) UOpBuilder {
        return .{ .allocator = alloc, .list = std.ArrayList(UOp).init(alloc) };
    }

    pub fn deinit(self: *UOpBuilder) void {
        // Free the allocated src slices before deiniting the list
        for (self.list.items) |uop| {
            if (uop.src.len > 0) {
                self.allocator.free(uop.src);
            }
        }
        self.list.deinit();
    }

    pub fn push(self: *UOpBuilder, op: UOpType, dtype: DType, src: []const usize, arg: ?Any) usize {
        const id = self.next_id;
        self.next_id += 1;
        // Allocate a copy of the src slice
        var owned_src: []usize = &[_]usize{};
        if (src.len > 0) {
            owned_src = self.allocator.alloc(usize, src.len) catch unreachable;
            @memcpy(owned_src, src);
        }
        // Cast to const slice when appending
        self.list.append(.{ .op = op, .dtype = dtype, .src = owned_src, .arg = arg }) catch unreachable;
        return id;
    }

    /// Transfers ownership of the internal UOp list (and its src slices) to the caller
    /// by creating a new owned slice containing only the used items.
    /// The builder should not be used after calling this.
    pub fn finalize(self: *UOpBuilder) ![]UOp {
        const owned_slice = try self.list.toOwnedSlice();
        // Deinit the now-empty internal list to free the original (potentially larger) buffer
        self.list.deinit();
        return owned_slice;
    }
};

//////////////////////////////////////////////////////////////////////////
// 2. HLOps Mocking Up The AST Linearized representation
//////////////////////////////////////////////////////////////////////////
pub const HLType = enum { Const, Add, Relu };

pub const HLOp = struct {
    typ: HLType,
    lhs: ?usize, // indices for inputs (Add, Relu)
    rhs: ?usize,
    val: f32 = 0, // scalar for Const
    elemN: usize = 0, // number of elements this op outputs
};

/// Returns the buffer-id (DEFINE_GLOBAL).
fn lowerConst(
    b: *UOpBuilder,
    value: f32,
) usize {
    // Just return the ID of the CONST op
    return b.push(.CONST, .f32, &[_]usize{}, Any{ .float = value });
}

fn lowerAdd(
    b: *UOpBuilder,
    lhs_buf: usize,
    rhs_buf: usize,
    out_buf: usize, // Added target output buffer ID
    N: usize,
) void { // Changed return type to void
    const range = b.push(.RANGE, .i32, &[_]usize{}, Any{ .loop_bounds = .{ .start = 0, .end = N } });

    const lhs = b.push(.LOAD, .f32, &[_]usize{ lhs_buf, range }, null);
    const rhs = b.push(.LOAD, .f32, &[_]usize{ rhs_buf, range }, null);
    const acc = b.push(.ADD, .f32, &[_]usize{ lhs, rhs }, null);

    _ = b.push(.STORE, .f32, &[_]usize{ out_buf, range, acc }, null); // Use out_buf
    _ = b.push(.ENDRANGE, .f32, &[_]usize{range}, null);
    // No return needed
}

fn lowerRelu(
    b: *UOpBuilder,
    in_buf: usize,
    out_buf: usize, // Added target output buffer ID
    N: usize,
) void { // Changed return type to void
    const loop = b.push(.RANGE, .i32, &[_]usize{}, Any{ .loop_bounds = .{ .start = 0, .end = N } });

    const x = b.push(.LOAD, .f32, &[_]usize{ in_buf, loop }, null);
    const z = b.push(.CONST, .f32, &[_]usize{}, Any{ .float = 0.0 }); // Use 0.0 float
    const f = b.push(.CMPLT, .bool, &[_]usize{ x, z }, null); // CMPLT output is bool
    const y = b.push(.WHERE, .f32, &[_]usize{ f, z, x }, null);
    _ = b.push(.STORE, .f32, &[_]usize{ out_buf, loop, y }, null); // Use out_buf

    _ = b.push(.ENDRANGE, .i32, &[_]usize{loop}, null);
    // No return needed
}

///////////////////////////////////////////////////////////////////////////////
// 4. Driver that consumes a linear HL list
///////////////////////////////////////////////////////////////////////////////
pub fn lowerHL(
    alloc: std.mem.Allocator,
    hl: []const HLOp,
    elemN: usize,
) ![]UOp {
    var b = UOpBuilder.init(alloc);
    // Defer deinit *before* finalizing, so finalize can take ownership
    defer b.deinit();

    // Allocate buffer IDs. We need one DEFINE_GLOBAL per HL op.
    var buf_ids = try alloc.alloc(usize, hl.len);
    defer alloc.free(buf_ids);
    for (0..hl.len) |i| {
        buf_ids[i] = b.push(.DEFINE_GLOBAL, .f32, &[_]usize{}, null);
    }

    // Track constant op IDs separately
    var const_op_ids = std.AutoHashMap(usize, usize).init(alloc);
    defer const_op_ids.deinit();

    for (hl, 0..) |node, idx| {
        switch (node.typ) {
            .Const => {
                // Store the actual CONST UOp ID associated with the HL Const index
                const const_id = lowerConst(&b, node.val);
                try const_op_ids.put(idx, const_id);
            },
            .Add => {
                // Get const IDs if inputs are Const, otherwise use buffer IDs
                const lhs_id = const_op_ids.get(node.lhs.?) orelse buf_ids[node.lhs.?];
                const rhs_id = const_op_ids.get(node.rhs.?) orelse buf_ids[node.rhs.?];
                lowerAdd(&b, lhs_id, rhs_id, buf_ids[idx], elemN);
            },
            .Relu => {
                const input_id = const_op_ids.get(node.lhs.?) orelse buf_ids[node.lhs.?];
                lowerRelu(&b, input_id, buf_ids[idx], elemN);
            },
        }
    }

    // Finalize the builder to get ownership of the UOp list
    return b.finalize();
}

pub const ZigRenderer = struct {
    // Function that returns the generic struct type
    pub fn new(comptime W: type) type {
        return struct { // The actual generic struct
            sb: W,
            names: std.AutoHashMap(usize, []const u8),

            // ----- ctor ----------------------------------------------------------
            pub fn init(a: *std.mem.Allocator, sb: W) @This() { // Return @This()
                return .{ .sb = sb, .names = .init(a.*) }; // Dereference allocator for HashMap
            }

            // ----- entry point ---------------------------------------------------
            pub fn render(self: *@This(), prog: []const UOp) !void { // Use @This()
                try self.renderPrologue();
                for (prog, 0..) |u, id| try self.renderOp(u, id); // Pass id
                try self.renderEpilogue();
            }

            // ----- dispatcher ----------------------------------------------------
            fn renderOp(self: *@This(), u: UOp, id: usize) !void { // Accept id
                switch (u.op) {
                    .DEFINE_GLOBAL => try self.renderDefineGlobal(u, id),
                    .CONST => try self.renderConst(u, id),
                    .RANGE => try self.renderRange(u, id),
                    .ENDRANGE => try self.renderEndRange(),
                    .LOAD => try self.renderLoad(u, id),
                    .STORE => try self.renderStore(u),
                    .ADD => try self.renderAdd(u, id),
                    .CMPLT => try self.renderCmpLt(u, id),
                    .WHERE => try self.renderWhere(u, id),
                    else => {}, // unhandled for brevity
                }
            }

            // ----- helpers to allocate unique variable names ---------------------
            fn fresh(self: *@This(), prefix: []const u8, id: usize) ![]const u8 { // Use @This()
                return std.fmt.allocPrint(self.names.allocator, "{s}{d}", .{ prefix, id });
            }

            fn nameOf(self: *@This(), id: usize) []const u8 { // Use @This()
                return self.names.get(id).?;
            }

            // ----- deinit --------------------------------------------------------
            pub fn deinit(self: *@This()) void {
                var it = self.names.valueIterator();
                while (it.next()) |name_ptr| {
                    self.names.allocator.free(name_ptr.*);
                }
                self.names.deinit();
            }

            // ----- prologue / epilogue ------------------------------------------
            fn renderPrologue(self: *@This()) !void { // Use @This()
                try self.sb.writeAll(
                    \\const std = @import("std");
                    \\
                    \\pub fn kernel(a: []const f32, b: []const f32, out: []f32) void {
                    \\
                );
            }
            fn renderEpilogue(self: *@This()) !void { // Use @This()
                try self.sb.writeAll("}\n");
            }

            // ----- concrete renderers -------------------------------------------
            fn renderDefineGlobal(self: *@This(), u: UOp, id: usize) !void { // Use @This()
                _ = u; // Mark u as used
                const name = try self.fresh("buf", id);
                _ = try self.names.put(id, name);
                // first 3 bufs correspond to a, b, out → no code needed
                if (id >= 3)
                    try self.sb.print("    var {s}: []f32 = undefined;\n", .{name});
            }

            fn renderConst(self: *@This(), u: UOp, id: usize) !void { // Use @This()
                const cname = try self.fresh("c", id);
                _ = try self.names.put(id, cname);
                const val = u.arg.?.float;
                // Use {d:.1} for simpler float formatting (e.g., 5.0)
                try self.sb.print("    const {s}: f32 = {d:.1};\n", .{ cname, val });
            }

            fn renderRange(self: *@This(), u: UOp, id: usize) !void { // Use @This()
                const iVar = try self.fresh("i", id);
                _ = try self.names.put(id, iVar);
                const lb = u.arg.?.loop_bounds.start;
                const ub = u.arg.?.loop_bounds.end;
                try self.sb.print("    for ({d}..{d}) |{s}| {{\n", .{ lb, ub, iVar });
            }

            fn renderEndRange(self: *@This()) !void { // Use @This()
                try self.sb.writeAll("    }\n");
            }

            fn renderLoad(self: *@This(), u: UOp, id: usize) !void { // Use @This()
                const dst = try self.fresh("t", id);
                _ = try self.names.put(id, dst);
                const src_id = u.src[0];
                const src_name = self.nameOf(src_id);

                // Check if loading from a const (name starts with 'c') or buffer
                if (std.mem.startsWith(u8, src_name, "c")) {
                    // Loading a constant - just assign the value
                    try self.sb.print("        const {s} = {s};\n", .{ dst, src_name });
                } else {
                    // Loading from a buffer - need index
                    const idx_name = self.nameOf(u.src[1]);
                    try self.sb.print("        const {s} = {s}[{s}];\n", .{ dst, src_name, idx_name });
                }
            }

            fn renderStore(self: *@This(), u: UOp) !void { // Removed id
                const target_buf_id = u.src[0];
                const idx = self.nameOf(u.src[1]);
                const val = self.nameOf(u.src[2]);

                // Map special buffer IDs to kernel parameters
                const target_name = switch (target_buf_id) {
                    0 => "a",
                    1 => "b",
                    2 => "out",
                    else => self.nameOf(target_buf_id),
                };

                try self.sb.print("        {s}[{s}] = {s};\n", .{ target_name, idx, val });
            }

            fn renderAdd(self: *@This(), u: UOp, id: usize) !void { // Use @This()
                const res = try self.fresh("t", id);
                _ = try self.names.put(id, res);
                const a = self.nameOf(u.src[0]);
                const b = self.nameOf(u.src[1]);
                try self.sb.print("        const {s} = {s} + {s};\n", .{ res, a, b });
            }

            fn renderCmpLt(self: *@This(), u: UOp, id: usize) !void { // Use @This()
                const res = try self.fresh("f", id);
                _ = try self.names.put(id, res);
                const a = self.nameOf(u.src[0]);
                const b = self.nameOf(u.src[1]);
                try self.sb.print("        const {s} = {s} < {s};\n", .{ res, a, b });
            }

            fn renderWhere(self: *@This(), u: UOp, id: usize) !void { // Use @This()
                const res = try self.fresh("t", id);
                _ = try self.names.put(id, res);
                const cond = self.nameOf(u.src[0]);
                const a = self.nameOf(u.src[1]);
                const b = self.nameOf(u.src[2]);
                try self.sb.print("        const {s} = if ({s}) {s} else {s};\n", .{ res, cond, a, b });
            }
        };
    }
};
