const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const Any = zant.uops.Any;

const MemoryRender = @import("memory_render.zig");
const ArithmeticRender = @import("arithmetic_render.zig");
const ReduceRender = @import("reduce_render.zig");
const ConditionalRender = @import("conditional_render.zig");
const UnaryRender = @import("unary_render.zig");
const GepRender = @import("gep_render.zig");
const ControlFlowRender = @import("controlflow_render.zig");
const ViewManagerModule = @import("view_manager.zig");
const ViewInfo = ViewManagerModule.ViewInfo;

const Uops = zant.uops;
const DTypeInfo = Uops.DTypeInfo;
const DType = Uops.DType;

// NEW: Structure to hold buffer info (inputs/outputs)
pub const BufferInfo = struct {
    id: usize,
    name: []const u8, // Note: lifetime managed by buffer_map allocator
    is_input: bool,
    dtype: DType,
    shape: []const usize, // Note: lifetime managed by buffer_map allocator
    len: usize,

    // Deinit is simple as allocator owns name/shape copies
};

pub fn ZigRenderer(comptime WriterType: type) type {
    return struct {
        allocator: std.mem.Allocator,
        writer: WriterType,
        rendered_ids: std.AutoHashMap(usize, void), // Changed to HashMap(usize, void)
        view_map: std.AutoHashMap(usize, ViewInfo),
        buffer_map: std.AutoHashMap(usize, BufferInfo),
        loop_indent: usize,
        final_output_buffer_id: ?usize = null, // Store ID of the presumed output buffer

        // Define local errors
        pub const RendererError = error{
            InvalidOperation,
            EmptyUOpList,
            InvalidLastUOp,
            OutputBufferNotFound,
            InputBufferNotFound,
            UnsupportedUOp, // Added this error
            VariableNotFound, // Added for missing IDs in ptr_map
            InvalidArgs, // Added for invalid args in render_mulacc
            OutOfMemory, // Added for memory allocation failures
        };

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, writer: WriterType) *Self {
            const self = allocator.create(Self) catch unreachable;
            self.* = Self{
                .allocator = allocator,
                .writer = writer,
                .rendered_ids = std.AutoHashMap(usize, void).init(allocator),
                .view_map = std.AutoHashMap(usize, ViewInfo).init(allocator),
                .buffer_map = std.AutoHashMap(usize, BufferInfo).init(allocator),
                .loop_indent = 0,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            // NEW: Free buffer info copies
            var iter = self.buffer_map.valueIterator();
            while (iter.next()) |info| {
                // Free the strings/slices allocated in identify_buffers
                self.allocator.free(info.name);
                self.allocator.free(info.shape);
            }

            self.buffer_map.deinit();
            self.rendered_ids.deinit();
            self.view_map.deinit();
            self.allocator.destroy(self);
        }

        // NEW: Helper to populate buffer_map
        fn identify_buffers(self: *Self, uops_list: []const UOp, input_ids: []const usize) !void {
            // Clear existing map (deinit frees old entries)
            var iter = self.buffer_map.valueIterator();
            while (iter.next()) |info| {
                self.allocator.free(info.name);
                self.allocator.free(info.shape);
            }

            self.buffer_map.clearRetainingCapacity();

            // Add explicit entries for input IDs
            for (input_ids) |in_id| {
                // We don't know shape/dtype from ID alone, need to find first usage?
                // Or assume they are passed externally. For now, create placeholder.
                // TODO: Find a way to get dtype/shape for inputs if needed here.
                const name = try std.fmt.allocPrint(self.allocator, "input_{d}", .{in_id});
                // Use getOrPut in case an input ID is also defined later (e.g., as DEFINE_GLOBAL)
                var entry = try self.buffer_map.getOrPut(in_id);
                if (!entry.found_existing) {
                    entry.value_ptr.* = BufferInfo{
                        .id = in_id,
                        .name = name,
                        .is_input = true,
                        .dtype = .f32, // Placeholder!
                        .shape = &[_]usize{}, // Placeholder!
                        .len = 0, // Initialize len
                    };
                } else {
                    // Input ID already exists (maybe from a later DEFINE_GLOBAL)
                    // Keep existing entry but mark as input and potentially update name.
                    // If the existing name wasn't an input name, free it.
                    if (!entry.value_ptr.is_input and !std.mem.startsWith(u8, entry.value_ptr.name, "input_")) {
                        self.allocator.free(entry.value_ptr.name);
                        entry.value_ptr.name = name; // Use the new input name
                    } else {
                        self.allocator.free(name); // Free the newly allocated name
                    }
                    entry.value_ptr.is_input = true;
                }
            }

            // Add/overwrite entries for ALL defining UOps
            for (uops_list) |uop| {
                // Check if it's an input ID (needed for name generation)
                const is_an_input = for (input_ids) |in_id| {
                    if (uop.id == in_id) break true;
                } else false;

                // Determine name prefix based on UOp type (keep this for ptr_map)
                const name_prefix = switch (uop.op) {
                    .DEFINE_GLOBAL => if (is_an_input) "input_" else "output_",
                    .GEP => "addr_", // Keep for GEP results
                    .RANGE => "idx_", // Keep for loop indices
                    .DEFINE_ACC => "acc_", // Keep for accumulators
                    else => "buf_", // Keep default for other values (LOAD, CONST, ALU)
                };

                const name = try std.fmt.allocPrint(self.allocator, "{s}{d}", .{ name_prefix, uop.id });
                var shape_copy: []const usize = &[_]usize{};
                var needs_free_shape = false;

                // Shape is only relevant for DEFINE_GLOBAL
                if (uop.op == .DEFINE_GLOBAL) {
                    if (uop.arg) |arg| {
                        if (arg == .shape) {
                            shape_copy = try self.allocator.dupe(usize, arg.shape);
                            needs_free_shape = shape_copy.len > 0;
                        }
                    }

                    if (!is_an_input) {
                        self.final_output_buffer_id = uop.id;
                    }
                }

                // Use getOrPut to add/update the entry for name mapping, but allocation happens elsewhere
                const entry = try self.buffer_map.getOrPut(uop.id);

                // Assign BufferInfo - NOTE: only DEFINE_GLOBAL info is fully used for allocation now
                if (!entry.found_existing) {
                    entry.value_ptr.* = BufferInfo{
                        .id = uop.id,
                        .name = name,
                        .is_input = is_an_input,
                        .dtype = uop.dtype,
                        .shape = shape_copy,
                        .len = 0,
                    };
                } else {
                    // Free resources if not needed
                    self.allocator.free(name);
                    if (needs_free_shape) self.allocator.free(shape_copy);
                }
            }
        }

        // UNCOMMENTING OLD RENDER METHOD
        pub fn render(self: *@This(), uops: []UOp) !void {
            self.rendered_ids.clearRetainingCapacity();
            for (uops) |uop| {
                // Use the new hasRendered logic
                if (self.hasRendered(uop.id)) continue;
                // NOTE: This old render path needs updating for ptr_map etc.
                // It likely won't produce correct pointer-based code for GEP/LOAD/STORE.
                // Let's call the new render_uop, but need a ptr_map...
                // For now, this old function is likely broken without a proper ptr_map.
                // Consider removing or updating it if needed.
                // Example placeholder call (will error without ptr_map):
                // var dummy_ptr_map = std.AutoHashMap(usize, []const u8).init(self.allocator);
                // defer dummy_ptr_map.deinit();
                // try self.render_uop(uop, &dummy_ptr_map);
                @panic("Old render function needs update for ptr_map");
                // Original switch removed as logic moves to render_uop
                // try self.rendered_ids.append(uop.id); // Use put for HashSet
            }
        }

        pub fn reset(self: *Self) void {
            self.buffer_map.clearRetainingCapacity();
            self.rendered_ids.clearRetainingCapacity();
            self.loop_indent = 0;
            self.final_output_buffer_id = null; // Reset output buffer ID
        }

        // NEW: Method to render a full function
        pub fn render_as_function(self: *Self, uops: []const UOp, input_ids: []const usize) !void {
            // Reset state
            self.reset();

            // 1. Identify buffers
            try self.identify_buffers(uops, input_ids);

            // 2. Get output buffer info
            const output_id = self.final_output_buffer_id orelse return error.OutputBufferNotFound;
            const output_info = self.buffer_map.get(output_id) orelse return error.OutputBufferNotFound;
            const output_type = DTypeInfo.asString(output_info.dtype);

            // 3. Generate function signature
            try self.write_function_signature(input_ids, output_type);

            // 4. Generate buffer allocations
            try self.write_buffer_allocations(output_id);

            // 5. Create ptr_map for kernel rendering
            var ptr_map = std.AutoHashMap(usize, []const u8).init(self.allocator);
            defer {
                self.free_ptr_map_values(&ptr_map);
                ptr_map.deinit();
            }

            // Populate ptr_map with buffer names
            var map_iter = self.buffer_map.iterator();
            while (map_iter.next()) |entry| {
                try ptr_map.put(entry.key_ptr.*, entry.value_ptr.name);
            }

            // 6. Render kernel body
            try self.render_kernel_body(uops, &ptr_map);

            // 7. Generate return statement
            try self.writer.print("    return {s};\n}}\n", .{output_info.name});
        }

        fn write_function_signature(self: *Self, input_ids: []const usize, output_type: []const u8) !void {
            try self.writer.print("pub fn generated_kernel(allocator: std.mem.Allocator", .{});

            // Add input parameters, sorted by ID
            var sorted_inputs = std.ArrayList(usize).init(self.allocator);
            defer sorted_inputs.deinit();
            try sorted_inputs.appendSlice(input_ids);
            std.mem.sort(usize, sorted_inputs.items, {}, std.sort.asc(usize));

            for (sorted_inputs.items) |id| {
                const info = self.buffer_map.get(id) orelse return RendererError.InputBufferNotFound;
                try self.writer.print(", {s}: []const {s}", .{ info.name, DTypeInfo.asString(info.dtype) });
            }

            // Return type
            try self.writer.print(") ![]{s} {{\n", .{output_type});
        }

        fn write_buffer_allocations(self: *Self, output_id: usize) !void {
            var alloc_iter = self.buffer_map.iterator();
            while (alloc_iter.next()) |entry| {
                const id = entry.key_ptr.*;
                const info = entry.value_ptr.*;

                // Skip inputs, loop vars, addresses, accumulators, and intermediate buffers
                if (info.is_input or
                    std.mem.startsWith(u8, info.name, "idx_") or
                    std.mem.startsWith(u8, info.name, "addr_") or
                    std.mem.startsWith(u8, info.name, "acc_") or
                    std.mem.startsWith(u8, info.name, "buf_"))
                {
                    continue;
                }

                // Calculate buffer size
                var size: usize = 1;
                if (info.shape.len > 0) {
                    for (info.shape) |dim| size *= dim;
                } else if (info.len > 0) {
                    size = info.len;
                }

                // Allocate buffer
                try self.writer.print("    const {s} = try allocator.alloc({s}, {d});\n", .{ info.name, DTypeInfo.asString(info.dtype), size });

                // Add defer free if not the output buffer
                if (id != output_id) {
                    try self.writer.print("    defer allocator.free({s});\n", .{info.name});
                }
            }
        }

        fn free_ptr_map_values(self: *Self, ptr_map: *std.AutoHashMap(usize, []const u8)) void {
            var ptr_map_value_iter = ptr_map.valueIterator();
            while (ptr_map_value_iter.next()) |var_name_ptr| {
                const var_name = var_name_ptr.*;
                // Free only dynamically allocated names not managed by buffer_map
                if (std.mem.startsWith(u8, var_name, "acc_") or
                    std.mem.startsWith(u8, var_name, "view_"))
                {
                    self.allocator.free(var_name);
                }
            }
        }

        fn render_kernel_body(self: *Self, uops: []const UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            for (uops) |uop| {
                if (uop.op == .DEFINE_GLOBAL) continue;
                if (self.rendered_ids.contains(uop.id)) continue;
                try self.render_uop(uop, ptr_map);
            }
        }

        // Helper to extract ID from names like "buf_123"
        fn getIdFromName(name: []const u8) usize {
            var parts = std.mem.splitScalar(u8, name, '_');
            if (parts.next() == null) return std.math.maxInt(usize); // Error case
            const id_str = parts.next() orelse return std.math.maxInt(usize);
            return std.fmt.parseInt(usize, id_str, 10) catch std.math.maxInt(usize);
        }

        fn hasRendered(self: *Self, id: usize) bool {
            // Use HashMap contains method
            return self.rendered_ids.contains(id);
        }

        // Helper to check if an ID is in the input list (used by identify_buffers)
        fn is_input(self: *Self, id: usize, input_ids: []const usize) bool {
            _ = self; // unused self ptr
            for (input_ids) |input_id| {
                if (id == input_id) return true;
            }
            return false;
        }

        // NEW: Helper function to render a single UOp
        fn render_uop(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            // Apply indentation based on loop depth
            try self.apply_indentation();

            switch (uop.op) {
                .DEFINE_GLOBAL, .LOAD, .STORE, .CONST => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),

                .GEP => try GepRender.render(self.allocator, self.writer, uop, &self.view_map, &self.buffer_map, ptr_map),

                .RANGE => {
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map);
                    self.loop_indent += 1;
                },

                .ENDRANGE => {
                    if (self.loop_indent > 0) self.loop_indent -= 1;
                    try self.apply_indentation();
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map);
                },

                .ADD, .SUB, .MUL, .FDIV, .MAX, .MIN, .CMPLT => try ArithmeticRender.render(self.allocator, self.writer, uop, ptr_map),

                .EXP2, .NEG, .CAST => try UnaryRender.render(self.allocator, self.writer, uop, ptr_map),

                .VIEW => try self.render_view(uop, ptr_map),
                .DEFINE_ACC => try self.render_define_acc(uop, ptr_map),
                .MULACC => try self.render_mulacc(uop, ptr_map),

                else => {
                    std.log.err("Rendering not implemented for UOp type: {s} (id: {d})\n", .{ @tagName(uop.op), uop.id });
                    return RendererError.UnsupportedUOp;
                },
            }

            // Mark UOp as rendered
            try self.rendered_ids.put(uop.id, {});
        }

        fn apply_indentation(self: *Self) !void {
            var i: usize = 0;
            while (i < self.loop_indent) : (i += 1) {
                try self.writer.print("    ", .{});
            }
        }

        fn render_view(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            // Record view metadata
            try ViewManagerModule.manage(uop, &self.view_map);

            // Create symbolic name for SSA id
            const view_name = try std.fmt.allocPrint(self.allocator, "view_{d}", .{uop.id});
            try ptr_map.put(uop.id, view_name);
        }

        fn render_define_acc(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            const dtype_str = DTypeInfo.asString(uop.dtype);
            const acc_name = try std.fmt.allocPrint(self.allocator, "acc_{d}", .{uop.id});
            // Ensure ptr_map owns the var_name slice
            try ptr_map.put(uop.id, acc_name);

            // Initialize accumulator to zero
            try self.writer.print("var {s}: {s} = 0;\n", .{ acc_name, dtype_str });
        }

        fn render_mulacc(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            if (uop.src.len != 3) return error.InvalidArgs;

            const acc_id = uop.src[0];
            const a_id = uop.src[1];
            const b_id = uop.src[2];

            // Get variable names from pointer map
            const acc_var = ptr_map.get(acc_id) orelse return RendererError.VariableNotFound;
            const a_var = ptr_map.get(a_id) orelse return RendererError.VariableNotFound;
            const b_var = ptr_map.get(b_id) orelse return RendererError.VariableNotFound;

            // Generate accumulation statement
            try self.writer.print("{s} += {s} * {s};\n", .{ acc_var, a_var, b_var });
        }
    };
}
