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
        };

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, writer: WriterType) *Self {
            const self = allocator.create(Self) catch unreachable;
            self.* = Self{
                .allocator = allocator,
                .writer = writer,
                .rendered_ids = std.AutoHashMap(usize, void).init(allocator), // Initialize HashMap
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
                var is_an_input = false;
                for (input_ids) |input_id| {
                    if (uop.id == input_id) {
                        is_an_input = true;
                        break;
                    }
                }

                // Generate name and shape based on op type
                var name: []const u8 = undefined;
                var shape_copy: []const usize = undefined;
                var needs_free_name = true; // Flag to track if generated name needs freeing
                var needs_free_shape_local: bool = false; // Revert to var with initialization

                // Determine name prefix based on UOp type (keep this for ptr_map)
                const name_prefix = switch (uop.op) {
                    .DEFINE_GLOBAL => if (is_an_input) "input_" else "output_",
                    .GEP => "addr_", // Keep for GEP results
                    .RANGE => "idx_", // Keep for loop indices
                    .DEFINE_ACC => "acc_", // Keep for accumulators
                    else => "buf_", // Keep default for other values (LOAD, CONST, ALU)
                };
                name = try std.fmt.allocPrint(self.allocator, "{s}{d}", .{ name_prefix, uop.id });

                // Shape is only relevant for DEFINE_GLOBAL
                if (uop.op == .DEFINE_GLOBAL) {
                    shape_copy = if (uop.arg) |arg| switch (arg) {
                        .shape => |s| try self.allocator.dupe(usize, s),
                        else => &[_]usize{}, // Placeholder
                    } else &[_]usize{}; // Placeholder
                    // Don't free static empty slice if shape is empty
                    needs_free_shape_local = shape_copy.len > 0; // Assign here
                } else {
                    shape_copy = &[_]usize{}; // No shape for intermediates
                    needs_free_shape_local = false; // Assign here
                    if (is_an_input) {
                        self.allocator.free(name);
                        needs_free_name = false;
                    }
                }

                // Use getOrPut to add/update the entry for name mapping, but allocation happens elsewhere
                const entry = try self.buffer_map.getOrPut(uop.id);

                // Assign BufferInfo - NOTE: only DEFINE_GLOBAL info is fully used for allocation now
                if (!entry.found_existing) {
                    entry.value_ptr.* = BufferInfo{
                        .id = uop.id,
                        .name = name, // Store generated name for ptr_map use
                        .is_input = is_an_input,
                        .dtype = uop.dtype,
                        .shape = shape_copy, // Store shape (mainly for DEFINE_GLOBAL)
                        .len = 0,
                    };
                    if (uop.op == .DEFINE_GLOBAL and !is_an_input) {
                        self.final_output_buffer_id = uop.id;
                    }
                } else {
                    // Existing entry found - potentially update if it was placeholder?
                    // Free newly generated name/shape if not needed
                    if (needs_free_name) self.allocator.free(name);
                    // Use variable declared in outer scope
                    if (needs_free_shape_local) self.allocator.free(shape_copy);
                    // Simplified update logic: Assume existing entry is sufficient unless specific cases arise
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

        // NEW: Method to render a full function
        pub fn render_as_function(self: *Self, uops: []const UOp, input_ids: []const usize) !void {
            // Reset state for new rendering
            self.buffer_map.clearRetainingCapacity();
            self.rendered_ids.clearRetainingCapacity();
            self.loop_indent = 0;
            self.final_output_buffer_id = null; // Reset output buffer ID
            var output_name_for_return: ?[]const u8 = null;
            var output_type_str: []const u8 = "void"; // Default to void

            // 1. Identify buffers and build buffer_map
            try self.identify_buffers(uops, input_ids);

            // --- NEW: Explicitly find the non-input DEFINE_GLOBAL for output ---
            // Get the output ID stored by identify_buffers
            const final_output_buffer_id = self.final_output_buffer_id orelse return error.OutputBufferNotFound; // Use stored ID

            // Get info for the identified output buffer
            const output_info = self.buffer_map.get(final_output_buffer_id) orelse return error.OutputBufferNotFound;
            output_name_for_return = output_info.name; // Assign name for return
            output_type_str = DTypeInfo.asString(output_info.dtype); // Assign type for signature

            // 2. Generate Function Signature using determined output type
            try self.writer.print("pub fn generated_kernel(allocator: std.mem.Allocator", .{});
            // Add input parameters to signature, sorted by ID for consistency
            var sorted_input_ids = std.ArrayList(usize).init(self.allocator);
            defer sorted_input_ids.deinit();
            try sorted_input_ids.appendSlice(input_ids);
            std.mem.sort(usize, sorted_input_ids.items, {}, std.sort.asc(usize));

            for (sorted_input_ids.items) |id| {
                const info = self.buffer_map.get(id) orelse return RendererError.InputBufferNotFound;
                // Assume inputs are const slices for now
                try self.writer.print(", {s}: []const {s}", .{ info.name, DTypeInfo.asString(info.dtype) });
            }
            // Return type is a slice of the output buffer's type
            try self.writer.print(") ![]{s} {{\n", .{output_type_str});

            // 3. Generate Allocations and Defers (excluding inputs and the final output)
            var alloc_iter = self.buffer_map.iterator();
            while (alloc_iter.next()) |entry| {
                const id = entry.key_ptr.*;
                const info = entry.value_ptr.*;
                // Skip inputs (already parameters)
                if (info.is_input) continue;
                // Skip loop variables (RANGE UOps)
                if (std.mem.startsWith(u8, info.name, "idx_")) continue;
                // Skip address variables (GEP UOps)
                if (std.mem.startsWith(u8, info.name, "addr_")) continue;
                // Skip accumulator variables (DEFINE_ACC UOps)
                if (std.mem.startsWith(u8, info.name, "acc_")) continue;
                // Skip intermediate buffer variables (LOAD, CONST, ALU UOps)
                if (std.mem.startsWith(u8, info.name, "buf_")) continue;

                // Allocate buffer (ONLY the final output buffer should remain)
                const type_str = DTypeInfo.asString(info.dtype);
                var size: usize = 1; // Calculate size based on shape
                if (info.shape.len > 0) { // Check if shape is available
                    for (info.shape) |dim| size *= dim;
                } else {
                    // Default size or error? Need to decide based on how shape info is populated
                    // For now, assume a default size might be needed if shape isn't always set
                    size = 1; // Placeholder: Get size from UOp if possible?
                    std.debug.print("WARN: Buffer {s} (id {d}) has no shape, allocating size 1\n", .{ info.name, id });
                    // Maybe use info.len if populated? Check identify_buffers
                    if (info.len > 0) size = info.len;
                }

                try self.writer.print("    const {s} = try allocator.alloc({s}, {d});\n", .{ info.name, type_str, size });

                // Add defer free ONLY if it's NOT the final output buffer
                if (id != final_output_buffer_id) {
                    try self.writer.print("    defer allocator.free({s});\n", .{info.name});
                }
            }

            // 4. Create Pointer Map for Kernel Rendering (maps buffer ID to its variable name)
            var ptr_map = std.AutoHashMap(usize, []const u8).init(self.allocator);
            // Defer deinit AFTER freeing allocated names
            defer ptr_map.deinit();

            var map_iter = self.buffer_map.iterator();
            while (map_iter.next()) |entry| {
                // Access key via ptr, value via ptr, then value's name field
                try ptr_map.put(entry.key_ptr.*, entry.value_ptr.name);
            }

            // 5. Render Kernel Body (UOps)
            for (uops) |uop| {
                if (uop.op == .DEFINE_GLOBAL) continue; // Skip globals in function body
                // Check if already rendered (using HashMap now)
                if (self.rendered_ids.contains(uop.id)) continue;

                // Call the dedicated render function for the UOp
                // Pass ptr_map by value, let render_uop modify it if needed (e.g., render_define_acc allocates name)
                try self.render_uop(uop, &ptr_map);
                // No need to mark rendered here, render_uop does it
            }

            // 6. Generate Return Statement using the identified output buffer name
            if (output_name_for_return) |name| {
                try self.writer.print("    return {s};\n", .{name});
            } else {
                // This case should ideally not be reached if output_buffer_id was found
                try self.writer.print("    // Error: No output buffer name found for return statement\n", .{});
            }

            // ptr_map.deinit() is deferred

            // Free names allocated specifically for ptr_map (e.g., by render_define_acc, render_gep)
            var ptr_map_value_iter = ptr_map.valueIterator();
            while (ptr_map_value_iter.next()) |var_name_ptr| {
                const var_name = var_name_ptr.*;
                // Free ONLY if it starts with a prefix indicating allocation during render
                // AND is NOT managed by buffer_map.
                // Currently, "acc_" and "view_" fit this criteria.
                if (std.mem.startsWith(u8, var_name, "acc_") or std.mem.startsWith(u8, var_name, "view_")) {
                    std.debug.print("DEBUG: Freeing allocated ptr_map value: {s}\n", .{var_name});
                    self.allocator.free(var_name);
                }
                // Note: "addr_" names ARE created in identify_buffers and stored in buffer_map,
                // so they are freed by ZigRenderer.deinit.
            }

            // 7. Closing Brace
            try self.writer.print("}}\n", .{});
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
            var i: usize = 0;
            while (i < self.loop_indent) : (i += 1) {
                try self.writer.print("    ", .{});
            }

            // Render based on UOp type
            switch (uop.op) {
                .DEFINE_GLOBAL => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .LOAD => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .STORE => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .GEP => {
                    try GepRender.render(self.allocator, self.writer, uop, &self.view_map, &self.buffer_map, ptr_map);
                },
                .RANGE => {
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map);
                    self.loop_indent += 1;
                },
                .ENDRANGE => {
                    // Dedent *before* printing the closing brace
                    if (self.loop_indent > 0) {
                        self.loop_indent -= 1;
                        // Re-apply indent for the closing brace itself
                        i = 0;
                        while (i < self.loop_indent) : (i += 1) {
                            try self.writer.print("    ", .{});
                        }
                    }
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map);
                },
                .ADD, .SUB, .MUL, .FDIV, .MAX, .MIN, .CMPLT => try ArithmeticRender.render(self.allocator, self.writer, uop, ptr_map),
                .EXP2, .NEG, .CAST => try UnaryRender.render(self.allocator, self.writer, uop, ptr_map),
                .CONST => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .VIEW => {
                    // 1. record the view's metadata
                    try ViewManagerModule.manage(uop, &self.view_map);

                    // 2. give the SSA id a symbolic name for ptr-map
                    //    (no code is emitted for VIEW itself)
                    const vname = try std.fmt.allocPrint(self.allocator, "view_{d}", .{uop.id});
                    try ptr_map.put(uop.id, vname);
                },

                .DEFINE_ACC => try self.render_define_acc(uop, ptr_map),
                .MULACC => try self.render_mulacc(uop, ptr_map),
                // TODO: Add rendering for other UOpTypes (REDUCE, CONDITIONAL, etc.)
                // .REDUCE_ADD, .REDUCE_MAX => try ReduceRender.render(...),
                // .IF, .ENDIF, .WHERE => try ConditionalRender.render(...),
                // .DEFINE_ACC, .MULACC => ??? (Need specific renderers)
                else => {
                    std.log.err("Rendering not implemented for UOp type: {s} (id: {d})\n", .{ @tagName(uop.op), uop.id });
                    return RendererError.UnsupportedUOp;
                },
            }

            // Mark UOp as rendered *after* successful rendering
            // Need putNoClobber or similar if an op could somehow be rendered twice validly?
            // Using put for now, assuming each ID corresponds to one render action.
            try self.rendered_ids.put(uop.id, {});
        }

        fn render_define_acc(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            const dtype_str = DTypeInfo.asString(uop.dtype);
            const var_name = try std.fmt.allocPrint(self.allocator, "acc_{d}", .{uop.id});
            // Ensure ptr_map owns the var_name slice
            try ptr_map.put(uop.id, var_name);

            // Initialize accumulator to zero
            try self.writer.print("var {s}: {s} = 0;\n", .{ var_name, dtype_str });
        }

        fn render_mulacc(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            // MULACC src = {acc_id, a_id, b_id}
            if (uop.src.len != 3) return error.InvalidArgs; // Assuming InvalidArgs error exists or should be added

            const acc_id = uop.src[0];
            const a_id = uop.src[1];
            const b_id = uop.src[2];

            // Use ptr_map.get() which returns an optional. Handle missing key.
            const acc_var = ptr_map.get(acc_id) orelse {
                std.log.err("Variable ID {d} not found in ptr_map for MULACC", .{acc_id});
                return RendererError.VariableNotFound;
            };
            const a_var = ptr_map.get(a_id) orelse {
                std.log.err("Variable ID {d} not found in ptr_map for MULACC", .{a_id});
                return RendererError.VariableNotFound;
            };
            const b_var = ptr_map.get(b_id) orelse {
                std.log.err("Variable ID {d} not found in ptr_map for MULACC", .{b_id});
                return RendererError.VariableNotFound;
            };

            // Generate: acc_var += a_var * b_var;
            try self.writer.print("{s} += {s} * {s};\n", .{ acc_var, a_var, b_var });
        }
    };
}
