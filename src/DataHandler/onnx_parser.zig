const std = @import("std");
const tensor = @import("tensor");
const Model = @import("model").Model;
const mem = std.mem;

const OnnxDataType = enum(i32) {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
};

pub const LayerParams = struct {
    name: ?[]const u8 = null,
    op_type: ?[]const u8 = null,
    kernel_shape: [2]usize = [_]usize{1} ** 2,
    strides: [2]usize = [_]usize{1} ** 2,
    padding: [4]usize = [_]usize{0} ** 4,
    groups: ?usize = 1,
    epsilon: ?f32 = null,
    perm: ?[]usize = null,
    axis: ?i64 = null,
    input_shape: ?[]usize = null,
    output_shape: ?[]usize = null,
    pool_type: ?[]const u8 = null,
    channels: ?[]usize = null,
    dilations: [2]usize = [_]usize{1} ** 2,
    auto_pad: ?[]const u8 = null, // NOTSET, SAME_UPPER, SAME_LOWER, VALID
};

const WireType = enum(u3) {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    StartGroup = 3,
    EndGroup = 4,
    Fixed32 = 5,
    Unknown6 = 6,
    Unknown7 = 7,
};

const BinaryReader = struct {
    data: []const u8,
    pos: usize,
    current_params: LayerParams = .{},
    current_attr_name: ?[]const u8 = null,

    pub fn init(data: []const u8) BinaryReader {
        return .{
            .data = data,
            .pos = 0,
        };
    }

    pub fn skipBytes(self: *BinaryReader, len: usize) !void {
        if (self.pos + len > self.data.len) {
            return error.EndOfBuffer;
        }
        self.pos += len;
    }

    pub fn readVarint(self: *BinaryReader) !u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        while (shift < 64) : (shift += 7) {
            if (self.pos >= self.data.len) return error.EndOfBuffer;
            const byte = self.data[self.pos];
            self.pos += 1;
            result |= @as(u64, byte & 0x7F) << shift;
            if (byte & 0x80 == 0) break;
        }
        if (shift >= 64) return error.VarintTooBig;
        return result;
    }

    const Tag = struct {
        field_number: u32,
        wire_type: WireType,
    };

    pub fn readTag(self: *BinaryReader) !Tag {
        const tag = try self.readVarint();
        return Tag{
            .field_number = @intCast(tag >> 3),
            .wire_type = @enumFromInt(@as(u3, @intCast(tag & 0x7))),
        };
    }

    pub fn readBytes(self: *BinaryReader) ![]const u8 {
        const len = try self.readVarint();
        if (len > std.math.maxInt(usize)) return error.LengthTooLarge;
        const size: usize = @intCast(len);
        if (self.pos + size > self.data.len) return error.EndOfBuffer;
        const bytes = self.data[self.pos..][0..size];
        self.pos += size;
        return bytes;
    }

    pub fn readString(self: *BinaryReader) ![]const u8 {
        const bytes = try self.readBytes();
        return bytes;
    }

    pub fn readFloat(self: *BinaryReader) !f32 {
        if (self.pos + 4 > self.data.len) return error.EndOfBuffer;
        const bytes = self.data[self.pos..][0..4];
        self.pos += 4;
        const int_value = bytes[0] | (@as(u32, bytes[1]) << 8) | (@as(u32, bytes[2]) << 16) | (@as(u32, bytes[3]) << 24);
        return @as(f32, @bitCast(int_value));
    }

    pub fn skipField(self: *BinaryReader, wire_type: WireType) !void {
        switch (wire_type) {
            .Varint => {
                _ = try self.readVarint();
            },
            .Fixed64 => {
                if (self.pos + 8 > self.data.len) return error.EndOfBuffer;
                self.pos += 8;
            },
            .LengthDelimited => {
                const len = try self.readVarint();
                if (len > std.math.maxInt(usize)) return error.LengthTooLarge;
                const size: usize = @intCast(len);
                if (self.pos + size > self.data.len) return error.EndOfBuffer;
                self.pos += size;
            },
            .StartGroup, .EndGroup => {
                // Groups are deprecated in proto3, but we'll skip them silently
                return;
            },
            .Fixed32 => {
                if (self.pos + 4 > self.data.len) return error.EndOfBuffer;
                self.pos += 4;
            },
            .Unknown6, .Unknown7 => {
                // Skip unknown wire types silently
                return;
            },
        }
    }

    pub fn readAttribute(self: *BinaryReader) !void {
        const tag = try self.readVarint();
        const wire_type = tag & 0x7;
        const field_number = tag >> 3;

        switch (field_number) {
            1 => { // name
                if (wire_type == 2) {
                    const name = try self.readString();
                    self.current_attr_name = name;
                }
            },
            2 => { // f
                if (wire_type == 1) {
                    if (self.pos + 8 > self.data.len) return error.EndOfBuffer;
                    const bytes = self.data[self.pos..][0..8];
                    self.pos += 8;
                    const value = @as(f32, @floatCast(@as(f64, @bitCast(bytes[0] |
                        (@as(u64, bytes[1]) << 8) |
                        (@as(u64, bytes[2]) << 16) |
                        (@as(u64, bytes[3]) << 24) |
                        (@as(u64, bytes[4]) << 32) |
                        (@as(u64, bytes[5]) << 40) |
                        (@as(u64, bytes[6]) << 48) |
                        (@as(u64, bytes[7]) << 56)))));

                    if (self.current_attr_name) |name| {
                        if (std.mem.eql(u8, name, "epsilon")) {
                            self.current_params.epsilon = value;
                        }
                    }
                }
            },
            3 => { // i64
                if (wire_type == 0) {
                    const value = try self.readVarint();
                    if (self.current_attr_name) |name| {
                        if (std.mem.eql(u8, name, "axis")) {
                            self.current_params.axis = @intCast(value);
                        }
                    }
                }
            },
            4 => { // ints
                if (wire_type == 2) {
                    const len = try self.readVarint();
                    if (self.current_attr_name) |name| {
                        var values = std.ArrayList(u64).init(std.heap.page_allocator);
                        defer values.deinit();

                        var i: usize = 0;
                        while (i < len) : (i += 1) {
                            const val = try self.readVarint();
                            try values.append(val);
                        }

                        if (std.mem.eql(u8, name, "kernel_shape")) {
                            if (values.items.len >= 2) {
                                self.current_params.kernel_shape[0] = @intCast(values.items[0]);
                                self.current_params.kernel_shape[1] = @intCast(values.items[1]);
                            }
                        } else if (std.mem.eql(u8, name, "strides")) {
                            if (values.items.len >= 2) {
                                self.current_params.strides[0] = @intCast(values.items[0]);
                                self.current_params.strides[1] = @intCast(values.items[1]);
                            }
                        } else if (std.mem.eql(u8, name, "pads")) {
                            if (values.items.len >= 4) {
                                self.current_params.padding[0] = @intCast(values.items[0]);
                                self.current_params.padding[1] = @intCast(values.items[1]);
                                self.current_params.padding[2] = @intCast(values.items[2]);
                                self.current_params.padding[3] = @intCast(values.items[3]);
                            }
                        } else if (std.mem.eql(u8, name, "dilations")) {
                            if (values.items.len >= 2) {
                                self.current_params.dilations[0] = @intCast(values.items[0]);
                                self.current_params.dilations[1] = @intCast(values.items[1]);
                            }
                        }
                    }
                }
            },
            7 => { // i
                if (wire_type == 0) {
                    const value = try self.readVarint();
                    if (self.current_attr_name) |name| {
                        if (std.mem.eql(u8, name, "group")) {
                            self.current_params.groups = @intCast(value);
                        }
                    }
                }
            },
            8 => { // repeated int64
                if (wire_type == 0) {
                    const value = try self.readVarint();
                    if (self.current_attr_name) |name| {
                        if (std.mem.eql(u8, name, "perm")) {
                            if (self.current_params.perm == null) {
                                var perm = std.ArrayList(usize).init(std.heap.page_allocator);
                                try perm.append(@intCast(value));
                                self.current_params.perm = try perm.toOwnedSlice();
                            } else {
                                const old_perm = self.current_params.perm.?;
                                var new_perm = std.ArrayList(usize).init(std.heap.page_allocator);
                                try new_perm.appendSlice(old_perm);
                                try new_perm.append(@intCast(value));
                                self.current_params.perm = try new_perm.toOwnedSlice();
                            }
                        }
                    }
                }
            },
            else => {
                try self.skipField(@enumFromInt(@as(u3, @intCast(wire_type))));
            },
        }
    }
};

pub fn parseOnnxFile(allocator: std.mem.Allocator, filepath: []const u8) !void {
    const file = try std.fs.cwd().openFile(filepath, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    if (file_size == 0) return error.EmptyFile;
    if (file_size > std.math.maxInt(usize)) return error.FileTooLarge;

    const buffer = try allocator.alloc(u8, @intCast(file_size));
    defer allocator.free(buffer);

    const bytes_read = try file.readAll(buffer);
    if (bytes_read != file_size) return error.ReadError;

    var reader = BinaryReader.init(buffer);
    var seen_layers = std.StringHashMap(void).init(allocator);
    defer seen_layers.deinit();

    std.debug.print("\n╔═══ ONNX Model Structure ═══╗\n", .{});
    std.debug.print("║                            ║\n", .{});
    var layer_count: usize = 0;
    var current_stage: ?[]const u8 = null;
    var current_block: ?[]const u8 = null;
    var last_block_num: ?i64 = null;
    var in_stem = false;
    var in_head = false;

    while (reader.pos < buffer.len) {
        const tag = reader.readTag() catch |err| switch (err) {
            error.EndOfBuffer => break,
            else => |e| return e,
        };

        switch (tag.wire_type) {
            .LengthDelimited => {
                const data = reader.readBytes() catch |err| switch (err) {
                    error.EndOfBuffer => continue,
                    else => |e| return e,
                };
                if (data.len == 0) continue;

                var node_reader = BinaryReader.init(data);
                while (node_reader.pos < data.len) {
                    const node_tag = node_reader.readTag() catch |err| switch (err) {
                        error.EndOfBuffer => break,
                        else => |e| return e,
                    };

                    switch (node_tag.wire_type) {
                        .LengthDelimited => {
                            const node_data = node_reader.readBytes() catch |err| switch (err) {
                                error.EndOfBuffer => continue,
                                else => |e| return e,
                            };
                            if (node_data.len == 0) continue;

                            if (isOperatorNode(node_data)) {
                                var params = LayerParams{};
                                processNode(node_data, &params);

                                if (params.op_type != null and params.name != null) {
                                    const name = params.name.?;
                                    if (!seen_layers.contains(name)) {
                                        try seen_layers.put(name, {});
                                        layer_count += 1;

                                        // Check for stem/head sections
                                        if (std.mem.indexOf(u8, name, "stem.")) |_| {
                                            if (!in_stem) {
                                                in_stem = true;
                                                std.debug.print("║ ┌─ Stem\n", .{});
                                            }
                                        } else if (std.mem.indexOf(u8, name, "head.")) |_| {
                                            if (!in_head) {
                                                in_head = true;
                                                std.debug.print("║ ┌─ Head\n", .{});
                                            }
                                        }

                                        // Check for stage transition
                                        if (std.mem.indexOf(u8, name, "stages.")) |_| {
                                            const stage = getStageNumber(name);
                                            if (stage) |s| {
                                                if (current_stage == null or !std.mem.eql(u8, current_stage.?, s)) {
                                                    current_stage = s;
                                                    last_block_num = null;
                                                    in_stem = false;
                                                    in_head = false;
                                                    std.debug.print("║\n║ ┌─ Stage {s}\n", .{s});
                                                }
                                            }
                                        }

                                        // Check for block transition
                                        if (std.mem.indexOf(u8, name, "blocks.")) |_| {
                                            const block = getBlockNumber(name);
                                            if (block) |b| {
                                                const block_num = std.fmt.parseInt(i64, b, 10) catch -1;
                                                if (last_block_num == null or block_num != last_block_num.?) {
                                                    current_block = b;
                                                    last_block_num = block_num;
                                                    std.debug.print("║ │  ┌─ Block {s}\n", .{b});
                                                }
                                            }
                                        }

                                        // Print layer info with parameters
                                        const clean_name = cleanupName(name);
                                        const part = getLayerPart(clean_name);
                                        const op_desc = getOperationDescription(params);

                                        // Print the main layer info with improved formatting
                                        if (std.mem.indexOf(u8, name, "blocks.")) |_| {
                                            std.debug.print("║ │  │  ", .{});
                                        } else if (std.mem.indexOf(u8, name, "stages.")) |_| {
                                            std.debug.print("║ │  ", .{});
                                        } else if (in_stem or in_head) {
                                            std.debug.print("║ │  ", .{});
                                        } else {
                                            std.debug.print("║ ", .{});
                                        }

                                        // Print layer name and type
                                        if (part) |p| {
                                            std.debug.print("└─ {s} ({s})", .{ clean_name, p });
                                        } else {
                                            std.debug.print("└─ {s}", .{clean_name});
                                        }

                                        // Print operation type and description
                                        std.debug.print(" → {s}", .{params.op_type.?});
                                        if (op_desc != null) {
                                            std.debug.print(" {s}", .{op_desc.?});
                                        }
                                        std.debug.print("\n", .{});

                                        // Close sections
                                        if (std.mem.indexOf(u8, name, "stem.")) |_| {
                                            if (std.mem.indexOf(u8, name, "stem.1.") != null) {
                                                std.debug.print("║ └────────────────\n", .{});
                                                in_stem = false;
                                            }
                                        } else if (std.mem.indexOf(u8, name, "head.")) |_| {
                                            if (std.mem.indexOf(u8, name, "head.1.") != null) {
                                                std.debug.print("║ └────────────────\n", .{});
                                                in_head = false;
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        else => node_reader.skipField(node_tag.wire_type) catch continue,
                    }
                }
            },
            else => reader.skipField(tag.wire_type) catch continue,
        }
    }

    std.debug.print("║\n║ Total layers: {d}\n", .{layer_count});
    std.debug.print("╚════════════════════════════╝\n", .{});
}

fn isOperatorNode(data: []const u8) bool {
    // Skip nodes that are clearly not operators
    if (std.mem.indexOf(u8, data, "bias") != null or
        std.mem.indexOf(u8, data, "weight") != null or
        std.mem.indexOf(u8, data, "running_mean") != null or
        std.mem.indexOf(u8, data, "running_var") != null or
        std.mem.indexOf(u8, data, "onnx::") != null)
    {
        return false;
    }

    // Check for layer operations
    // First check if it's a node name (contains a path-like structure)
    if (std.mem.indexOf(u8, data, "/") != null) {
        return true;
    }

    // Then check for specific operation types
    return std.mem.indexOf(u8, data, "Conv") != null or
        std.mem.indexOf(u8, data, "Pool") != null or
        std.mem.indexOf(u8, data, "Gemm") != null or
        std.mem.indexOf(u8, data, "Relu") != null or
        std.mem.indexOf(u8, data, "Add") != null or
        std.mem.indexOf(u8, data, "BatchNormalization") != null or
        std.mem.indexOf(u8, data, "Flatten") != null or
        std.mem.indexOf(u8, data, "Concat") != null or
        std.mem.indexOf(u8, data, "Sigmoid") != null or
        std.mem.indexOf(u8, data, "Softmax") != null or
        std.mem.indexOf(u8, data, "LayerNorm") != null or
        std.mem.indexOf(u8, data, "GELU") != null or
        std.mem.indexOf(u8, data, "Dropout") != null or
        std.mem.indexOf(u8, data, "MatMul") != null or
        std.mem.indexOf(u8, data, "Reshape") != null or
        std.mem.indexOf(u8, data, "Transpose") != null;
}

fn getInputShape(data: []const u8) ?[]usize {
    var reader = BinaryReader.init(data);
    var values = std.ArrayList(usize).init(std.heap.page_allocator);

    while (reader.pos < reader.data.len) {
        const tag = reader.readTag() catch return null;

        switch (tag.wire_type) {
            .LengthDelimited => {
                const bytes = reader.readBytes() catch return null;
                var value_info_reader = BinaryReader.init(bytes);

                while (value_info_reader.pos < value_info_reader.data.len) {
                    const value_tag = value_info_reader.readTag() catch break;

                    if (value_tag.field_number == 2) { // type field
                        const type_bytes = value_info_reader.readBytes() catch break;
                        var type_reader = BinaryReader.init(type_bytes);

                        while (type_reader.pos < type_reader.data.len) {
                            const tensor_tag = type_reader.readTag() catch break;

                            if (tensor_tag.field_number == 2) { // tensor_shape field
                                const shape_bytes = type_reader.readBytes() catch break;
                                var shape_reader = BinaryReader.init(shape_bytes);

                                while (shape_reader.pos < shape_reader.data.len) {
                                    const dim_tag = shape_reader.readTag() catch break;

                                    if (dim_tag.field_number == 2) { // dim_value field
                                        const val = shape_reader.readVarint() catch break;
                                        values.append(@intCast(val)) catch break;
                                    } else {
                                        shape_reader.skipField(dim_tag.wire_type) catch break;
                                    }
                                }
                            } else {
                                type_reader.skipField(tensor_tag.wire_type) catch break;
                            }
                        }
                    } else {
                        value_info_reader.skipField(value_tag.wire_type) catch break;
                    }
                }
            },
            else => reader.skipField(tag.wire_type) catch continue,
        }
    }

    if (values.items.len > 0) {
        return values.toOwnedSlice() catch return null;
    }
    return null;
}

fn getNodeName(data: []const u8) ?[]const u8 {
    // First try to find a clean layer name in the format "stages.X.blocks.Y..."
    if (std.mem.indexOf(u8, data, "stages.")) |start| {
        var end = start;
        while (end < data.len and !std.mem.containsAtLeast(u8, data[end..], 1, " /\n\r\t")) : (end += 1) {}
        if (end > start) {
            return data[start..end];
        }
    }

    // Then try to find stem layer
    if (std.mem.indexOf(u8, data, "stem.")) |start| {
        var end = start;
        while (end < data.len and !std.mem.containsAtLeast(u8, data[end..], 1, " /\n\r\t")) : (end += 1) {}
        if (end > start) {
            return data[start..end];
        }
    }

    // Then try to find head layer
    if (std.mem.indexOf(u8, data, "head.")) |start| {
        var end = start;
        while (end < data.len and !std.mem.containsAtLeast(u8, data[end..], 1, " /\n\r\t")) : (end += 1) {}
        if (end > start) {
            return data[start..end];
        }
    }

    // Finally try to find any clean text before special characters
    var end: usize = 0;
    while (end < data.len and !std.mem.containsAtLeast(u8, data[end..], 1, " /\n\r\t@\xFF")) : (end += 1) {}
    if (end > 0) {
        return data[0..end];
    }

    return null;
}

fn getStageNumber(name: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, name, "stages.")) |start| {
        const after_dot = start + "stages.".len;
        var end = after_dot;
        while (end < name.len and std.ascii.isDigit(name[end])) : (end += 1) {}
        if (end > after_dot) {
            return name[after_dot..end];
        }
    }
    return null;
}

fn getBlockNumber(name: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, name, "blocks.")) |start| {
        const after_dot = start + "blocks.".len;
        var end = after_dot;
        while (end < name.len and std.ascii.isDigit(name[end])) : (end += 1) {}
        if (end > after_dot) {
            return name[after_dot..end];
        }
    }
    return null;
}

fn getLayerPart(name: []const u8) ?[]const u8 {
    if (std.mem.lastIndexOf(u8, name, ".")) |last_dot| {
        const part = name[last_dot + 1 ..];
        if (!std.ascii.isDigit(part[0])) {
            return part;
        }
    }
    return null;
}

pub fn printModel(model_: *const Model(f32)) void {
    std.debug.print("\n=== ONNX Model Structure ===\n", .{});

    const layers = model_.layers.items;
    std.debug.print("Number of layers: {d}\n", .{layers.len});
    for (layers, 0..) |layer, i| {
        std.debug.print("\nLayer {d}:\n", .{i});
        std.debug.print("  Type: {s}\n", .{@tagName(layer.layer_type)});
        std.debug.print("  Inputs: {d}\n", .{layer.layer_impl.get_n_inputs(layer.layer_ptr)});
        std.debug.print("  Neurons: {d}\n", .{layer.layer_impl.get_n_neurons(layer.layer_ptr)});

        if (layer.layer_impl.get_input(layer.layer_ptr)) |input| {
            std.debug.print("  Input shape: ", .{});
            for (input.shape) |dim| {
                std.debug.print("{d} ", .{dim});
            }
            std.debug.print("\n", .{});
        }

        if (layer.layer_impl.get_output(layer.layer_ptr)) |output| {
            std.debug.print("  Output shape: ", .{});
            for (output.shape) |dim| {
                std.debug.print("{d} ", .{dim});
            }
            std.debug.print("\n", .{});
        }
    }

    std.debug.print("=========================\n", .{});
}

fn cleanupName(name: []const u8) []const u8 {
    // Remove output node suffixes
    if (std.mem.indexOf(u8, name, "_output_0")) |idx| {
        return name[0..idx];
    }

    // Find the first occurrence of special characters
    for (name, 0..) |c, i| {
        if (c == '/' or c == '"' or c == '@' or c == 0xFF or c == '\n' or c == '\r' or c == '\t') {
            return name[0..i];
        }
    }
    return name;
}

fn getOperationDescription(params: LayerParams) ?[]const u8 {
    if (params.op_type) |op| {
        var desc = std.ArrayList(u8).init(std.heap.page_allocator);
        defer desc.deinit();

        if (std.mem.eql(u8, op, "Conv")) {
            desc.writer().print("[", .{}) catch return null;
            var first = true;
            if (params.kernel_shape[0] != 1 or params.kernel_shape[1] != 1) {
                if (!first) desc.writer().print(" ", .{}) catch return null;
                desc.writer().print("k={d}x{d}", .{ params.kernel_shape[0], params.kernel_shape[1] }) catch return null;
                first = false;
            }
            if (params.strides[0] != 1 or params.strides[1] != 1) {
                if (!first) desc.writer().print(" ", .{}) catch return null;
                desc.writer().print("s={d}x{d}", .{ params.strides[0], params.strides[1] }) catch return null;
                first = false;
            }
            if (params.padding[0] != 0 or params.padding[1] != 0 or params.padding[2] != 0 or params.padding[3] != 0) {
                if (!first) desc.writer().print(" ", .{}) catch return null;
                desc.writer().print("p={d},{d},{d},{d}", .{ params.padding[0], params.padding[1], params.padding[2], params.padding[3] }) catch return null;
                first = false;
            }
            if (params.dilations[0] != 1 or params.dilations[1] != 1) {
                if (!first) desc.writer().print(" ", .{}) catch return null;
                desc.writer().print("d={d}x{d}", .{ params.dilations[0], params.dilations[1] }) catch return null;
                first = false;
            }
            if (params.groups != null and params.groups.? != 1) {
                if (!first) desc.writer().print(" ", .{}) catch return null;
                desc.writer().print("g={d}", .{params.groups.?}) catch return null;
                first = false;
            }
            desc.writer().print("]", .{}) catch return null;
        } else if (std.mem.eql(u8, op, "LayerNormalization")) {
            if (params.epsilon != null) {
                desc.writer().print("[ε={d:.6}]", .{params.epsilon.?}) catch return null;
            }
        } else if (std.mem.eql(u8, op, "Transpose")) {
            if (params.perm != null) {
                desc.writer().print("[", .{}) catch return null;
                for (params.perm.?, 0..) |p, i| {
                    if (i > 0) desc.writer().print(",", .{}) catch return null;
                    desc.writer().print("{d}", .{p}) catch return null;
                }
                desc.writer().print("]", .{}) catch return null;
            }
        } else if (std.mem.eql(u8, op, "Flatten")) {
            if (params.axis != null) {
                desc.writer().print("[axis={d}]", .{params.axis.?}) catch return null;
            }
        }

        if (desc.items.len > 0) {
            return desc.toOwnedSlice() catch return null;
        }
    }
    return null;
}

fn getOperatorType(data: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, data, "Conv") != null) return "Conv";
    if (std.mem.indexOf(u8, data, "Pool") != null) return "Pool";
    if (std.mem.indexOf(u8, data, "Gemm") != null) return "Linear";
    if (std.mem.indexOf(u8, data, "Relu") != null) return "ReLU";
    if (std.mem.indexOf(u8, data, "Add") != null) return "Add";
    if (std.mem.indexOf(u8, data, "BatchNormalization") != null) return "BatchNorm";
    if (std.mem.indexOf(u8, data, "Flatten") != null) return "Flatten";
    if (std.mem.indexOf(u8, data, "Concat") != null) return "Concat";
    if (std.mem.indexOf(u8, data, "Sigmoid") != null) return "Sigmoid";
    if (std.mem.indexOf(u8, data, "Softmax") != null) return "Softmax";
    if (std.mem.indexOf(u8, data, "LayerNorm") != null) return "LayerNorm";
    if (std.mem.indexOf(u8, data, "GELU") != null) return "GELU";
    if (std.mem.indexOf(u8, data, "Dropout") != null) return "Dropout";
    if (std.mem.indexOf(u8, data, "MatMul") != null) return "MatMul";
    if (std.mem.indexOf(u8, data, "Reshape") != null) return "Reshape";
    if (std.mem.indexOf(u8, data, "Transpose") != null) return "Transpose";
    return null;
}

fn processNode(data: []const u8, params: *LayerParams) void {
    var reader = BinaryReader.init(data);

    while (reader.pos < data.len) {
        const tag = reader.readTag() catch break;

        switch (tag.field_number) {
            1 => { // name
                params.name = reader.readString() catch break;
            },
            4 => { // op_type
                params.op_type = reader.readString() catch break;
            },
            5 => { // attribute
                const attr_data = reader.readBytes() catch break;
                var attr_parser = BinaryReader.init(attr_data);

                while (attr_parser.pos < attr_parser.data.len) {
                    const field_tag = attr_parser.readTag() catch break;

                    switch (field_tag.field_number) {
                        1 => { // name
                            const attr_name = attr_parser.readString() catch break;

                            // Skip type field
                            _ = attr_parser.readTag() catch break;
                            _ = attr_parser.readVarint() catch break;

                            std.debug.print("\n╭─ Layer Attributes ─────────────\n", .{});
                            if (params.name != null) {
                                std.debug.print("│ Name: {s}\n", .{params.name.?});
                            }
                            if (params.op_type != null) {
                                std.debug.print("│ Type: {s}\n", .{params.op_type.?});
                            }
                            std.debug.print("│ Attribute: {s}\n", .{attr_name});

                            if (std.mem.eql(u8, attr_name, "auto_pad")) {
                                const str_tag = attr_parser.readTag() catch break;
                                if (str_tag.field_number == 3) { // string
                                    const val = attr_parser.readString() catch break;
                                    params.auto_pad = val;
                                    std.debug.print("│ Value: {s}\n", .{val});
                                }
                            } else if (std.mem.eql(u8, attr_name, "kernel_shape")) {
                                const ints_tag = attr_parser.readTag() catch break;
                                if (ints_tag.field_number == 7) { // ints
                                    const len = attr_parser.readVarint() catch break;
                                    var i: usize = 0;
                                    std.debug.print("│ Values: ", .{});
                                    while (i < len and i < 2) : (i += 1) {
                                        const val = attr_parser.readVarint() catch break;
                                        params.kernel_shape[i] = @intCast(val);
                                        std.debug.print("{d} ", .{val});
                                    }
                                    std.debug.print("\n", .{});
                                }
                            } else if (std.mem.eql(u8, attr_name, "strides")) {
                                const ints_tag = attr_parser.readTag() catch break;
                                if (ints_tag.field_number == 7) { // ints
                                    const len = attr_parser.readVarint() catch break;
                                    var i: usize = 0;
                                    std.debug.print("│ Values: ", .{});
                                    while (i < len and i < 2) : (i += 1) {
                                        const val = attr_parser.readVarint() catch break;
                                        params.strides[i] = @intCast(val);
                                        std.debug.print("{d} ", .{val});
                                    }
                                    std.debug.print("\n", .{});
                                }
                            } else if (std.mem.eql(u8, attr_name, "pads")) {
                                const ints_tag = attr_parser.readTag() catch break;
                                if (ints_tag.field_number == 7) { // ints
                                    const len = attr_parser.readVarint() catch break;
                                    var i: usize = 0;
                                    std.debug.print("│ Values: ", .{});
                                    while (i < len and i < 4) : (i += 1) {
                                        const val = attr_parser.readVarint() catch break;
                                        params.padding[i] = @intCast(val);
                                        std.debug.print("{d} ", .{val});
                                    }
                                    std.debug.print("\n", .{});
                                }
                            } else if (std.mem.eql(u8, attr_name, "group")) {
                                const int_tag = attr_parser.readTag() catch break;
                                if (int_tag.field_number == 2) { // i
                                    const val = attr_parser.readVarint() catch break;
                                    params.groups = @intCast(val);
                                    std.debug.print("│ Value: {d}\n", .{val});
                                }
                            } else if (std.mem.eql(u8, attr_name, "dilations")) {
                                const ints_tag = attr_parser.readTag() catch break;
                                if (ints_tag.field_number == 7) { // ints
                                    const len = attr_parser.readVarint() catch break;
                                    var i: usize = 0;
                                    std.debug.print("│ Values: ", .{});
                                    while (i < len and i < 2) : (i += 1) {
                                        const val = attr_parser.readVarint() catch break;
                                        params.dilations[i] = @intCast(val);
                                        std.debug.print("{d} ", .{val});
                                    }
                                    std.debug.print("\n", .{});
                                }
                            } else {
                                // Print unknown attribute values
                                const next_tag = attr_parser.readTag() catch break;
                                if (next_tag.field_number == 2) { // int
                                    const val = attr_parser.readVarint() catch break;
                                    std.debug.print("│ Value (int): {d}\n", .{val});
                                } else if (next_tag.field_number == 3) { // string
                                    const val = attr_parser.readString() catch break;
                                    std.debug.print("│ Value (string): {s}\n", .{val});
                                } else if (next_tag.field_number == 7) { // ints
                                    const len = attr_parser.readVarint() catch break;
                                    var i: usize = 0;
                                    std.debug.print("│ Values (ints): ", .{});
                                    while (i < len) : (i += 1) {
                                        const val = attr_parser.readVarint() catch break;
                                        std.debug.print("{d} ", .{val});
                                    }
                                    std.debug.print("\n", .{});
                                }
                            }
                            std.debug.print("╰───────────────────────────────────\n", .{});
                        },
                        else => attr_parser.skipField(field_tag.wire_type) catch break,
                    }
                }
            },
            else => reader.skipField(tag.wire_type) catch break,
        }
    }
}
