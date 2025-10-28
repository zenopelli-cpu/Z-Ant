const std = @import("std");
const protobuf = @import("protobuf.zig");
const fromString = @import("onnx.zig").fromString;
const AttributeType = @import("onnx.zig").AttributeType;
const AttributeProto = @import("onnx.zig").AttributeProto;
const DataType = @import("onnx.zig").DataType;
const OnnxOperator = @import("onnx.zig").OnnxOperator;
const StringStringEntryProto = @import("onnx.zig").StringStringEntryProto;

//--
const parseError = @import("parseErrors.zig");
//--

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

const onnx_log = std.log.scoped(.nodeProto);

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
    op_type: OnnxOperator,
    domain: ?[]const u8,
    input: [][]const u8,
    output: [][]const u8,
    attribute: []*AttributeProto,
    doc_string: ?[]const u8,
    overload: ?[]const u8,
    metadata_props: []*StringStringEntryProto,

    pub fn deinit(self: *NodeProto, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);
        //allocator.free(self.op_type); TODO remove this line because it is useless
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
        if (self.doc_string) |doc_string| allocator.free(doc_string);
        if (self.overload) |overload| allocator.free(overload);
        allocator.free(self.metadata_props);
    }

    pub fn parse(reader: *protobuf.ProtoReader) parseError.ParseError!NodeProto {
        var node = NodeProto{
            .name = null,
            .op_type = undefined,
            .domain = null,
            .input = &[_][]const u8{},
            .output = &[_][]const u8{},
            .attribute = &[_]*AttributeProto{},
            .doc_string = null,
            .overload = null,
            .metadata_props = undefined,
        };

        var inputs: std.ArrayList([]const u8) = .empty;
        defer inputs.deinit(reader.allocator);
        var outputs: std.ArrayList([]const u8) = .empty;
        defer outputs.deinit(reader.allocator);
        var attributes: std.ArrayList(*AttributeProto) = .empty;
        defer attributes.deinit(reader.allocator);
        var metadataList: std.ArrayList(*StringStringEntryProto) = .empty;
        defer metadataList.deinit(reader.allocator);

        errdefer {
            if (node.name) |n| reader.allocator.free(n);
            if (node.domain) |d| reader.allocator.free(d);
            // reader.allocator.free(node.op_type); //TODO delte this line because it is useless

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
                    try inputs.append(reader.allocator, value);
                },
                2 => { // output
                    const value = try reader.readString(reader.allocator);
                    try outputs.append(reader.allocator, value);
                },
                3 => { // name
                    node.name = try reader.readString(reader.allocator);
                },
                4 => { // op_type
                    const op_str = try reader.readString(reader.allocator);
                    node.op_type = try fromString(op_str);
                    reader.allocator.free(op_str);
                },
                5 => { // attribute (repeated)
                    var attr_reader = try reader.readLengthDelimited();
                    const attr_ptr = try reader.allocator.create(AttributeProto);
                    attr_ptr.* = try AttributeProto.parse(&attr_reader);
                    try attributes.append(reader.allocator, attr_ptr);
                },
                6 => { // doc_string
                    node.doc_string = try reader.readString(reader.allocator);
                },
                7 => { // domain
                    node.domain = try reader.readString(reader.allocator);
                },
                8 => { //overload
                    node.overload = try reader.readString(reader.allocator);
                },
                9 => { // metadata_props
                    var md_reader = try reader.readLengthDelimited(); //var md_reader
                    const ssep_ptr = try reader.allocator.create(StringStringEntryProto);
                    ssep_ptr.* = try StringStringEntryProto.parse(&md_reader);
                    try metadataList.append(reader.allocator, ssep_ptr);
                },
                else => {
                    onnx_log.warn("\n\n ERROR: tag{} NOT AVAILABLE for NodeProto\n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        node.input = try inputs.toOwnedSlice(reader.allocator);
        node.output = try outputs.toOwnedSlice(reader.allocator);
        node.attribute = try attributes.toOwnedSlice(reader.allocator);
        node.metadata_props = try metadataList.toOwnedSlice(reader.allocator);
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

        std.debug.print("{s}Op Type: {any}\n", .{ space, self.op_type });

        if (self.domain) |d| {
            std.debug.print("{s}Domain: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}Domain: (none)\n", .{space});
        }

        std.debug.print("{s}Inputs: ", .{space});
        for (
            self.input,
        ) |inp| {
            std.debug.print("{s}  -  ", .{if (std.mem.eql(u8, inp, "")) "<empty_string>" else inp});
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Outputs: ", .{space});
        for (self.output) |out| {
            std.debug.print("{s}{s} ", .{ space, out });
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Attributes:\n", .{space});
        for (self.attribute) |attr| {
            attr.print(space);
        }

        std.debug.print("{s}metadata_props (key, value) [{}]: \n", .{ space, self.metadata_props.len });
        for (self.metadata_props) |mp| {
            mp.print(space);
        }
    }
};
