const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const AttributeProto = @import("onnx.zig").AttributeProto;
const DataType = @import("onnx.zig").DataType;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

//onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L212
//TAGS:
//  - 1 : input, repeated string
//  - 2 : output, repeated string
//  - 3 : name, optional string
//  - 4 : op_type, optional string
//  - 5 : attribute, repeated AttributeProto
//  - 6 : doc_string, optional string
//  - 7 : domain, optional string
//  - 8 : overload, optional string
//  - 9 : metadata_props, repeated StringStringEntryProto
pub const NodeProto = struct {
    name: ?[]const u8,
    op_type: []const u8,
    domain: ?[]const u8,
    input: [][]const u8,
    output: [][]const u8,
    attribute: []*AttributeProto,

    pub fn deinit(self: *NodeProto, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);
        allocator.free(self.op_type);
        if (self.domain) |domain| allocator.free(domain);
        for (self.input) |input| {
            allocator.free(input);
        }
        allocator.free(self.input);
        for (self.output) |output| {
            allocator.free(output);
        }
        allocator.free(self.output);
        for (self.attribute) |attr| {
            attr.deinit(allocator);
            allocator.destroy(attr);
        }
        allocator.free(self.attribute);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !NodeProto {
        var node = NodeProto{
            .name = null,
            .op_type = undefined,
            .domain = null,
            .input = &[_][]const u8{},
            .output = &[_][]const u8{},
            .attribute = &[_]*AttributeProto{},
        };

        var inputs = std.ArrayList([]const u8).init(reader.allocator);
        defer inputs.deinit();
        var outputs = std.ArrayList([]const u8).init(reader.allocator);
        defer outputs.deinit();
        var attributes = std.ArrayList(*AttributeProto).init(reader.allocator);
        defer attributes.deinit();

        errdefer {
            if (node.name) |n| reader.allocator.free(n);
            if (node.domain) |d| reader.allocator.free(d);
            reader.allocator.free(node.op_type);

            for (inputs.items) |i| reader.allocator.free(i);
            for (outputs.items) |o| reader.allocator.free(o);
            for (attributes.items) |attr| {
                attr.deinit(reader.allocator);
                reader.allocator.destroy(attr);
            }
        }

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // input
                    const value = try reader.readString(reader.allocator);
                    try inputs.append(value);
                },
                2 => { // output
                    const value = try reader.readString(reader.allocator);
                    try outputs.append(value);
                },
                3 => { // name
                    node.name = try reader.readString(reader.allocator);
                },
                4 => { // op_type
                    node.op_type = try reader.readString(reader.allocator);
                },
                5 => { // attribute (repeated)
                    var attr_reader = try reader.readLengthDelimited();
                    const attr_ptr = try reader.allocator.create(AttributeProto);
                    attr_ptr.* = try AttributeProto.parseSingleAttribute(&attr_reader, reader.allocator);
                    try attributes.append(attr_ptr);
                },
                7 => { // domain
                    node.domain = try reader.readString(reader.allocator);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for NodeProto\n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        node.input = try inputs.toOwnedSlice();
        node.output = try outputs.toOwnedSlice();
        node.attribute = try attributes.toOwnedSlice();
        return node;
    }

    pub fn print(self: *NodeProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- NODE\n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Name: (none)\n", .{space});
        }

        std.debug.print("{s}Op Type: {s}\n", .{ space, self.op_type });

        if (self.domain) |d| {
            std.debug.print("{s}Domain: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}Domain: (none)\n", .{space});
        }

        std.debug.print("{s}Inputs: ", .{space});
        for (self.input, 0..) |inp, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{inp});
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Outputs: ", .{space});
        for (self.output, 0..) |out, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}{s} ", .{ space, out });
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Attributes:\n", .{space});
        for (self.attribute) |attr| {
            attr.print(space);
        }
    }
};
