const std = @import("std");
const protobuf = @import("protobuf.zig");
const TypeProto = @import("onnx.zig").TypeProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

//https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L193
//TAGS:
//  - 1 : name, string
//  - 2 : type, TypeProto
//  - 3 : doc_string, string
//  - 4 : TODO metadata_props, StringStringEntryProto
pub const ValueInfoProto = struct {
    name: ?[]const u8,
    type: ?*TypeProto,
    doc_string: ?[]const u8,

    pub fn deinit(self: *ValueInfoProto, allocator: std.mem.Allocator) void {
        if (self.name) |n| allocator.free(n);
        if (self.doc_string) |doc_string| allocator.free(doc_string);
        if (self.type) |t| {
            t.deinit(allocator);
            allocator.destroy(t);
        }
    }

    pub fn parse(reader: *protobuf.ProtoReader) !ValueInfoProto {
        var value_info = ValueInfoProto{
            .name = undefined,
            .type = undefined,
            .doc_string = undefined,
        };

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // name
                    std.debug.print("\n ................ ValueInfoProto READING name ", .{});
                    value_info.name = try reader.readString(reader.allocator);
                },
                2 => { // type
                    std.debug.print("\n ................ ValueInfoProto READING type ", .{});

                    var type_reader = try reader.readLengthDelimited(); //var type_reader
                    const type_ptr = try reader.allocator.create(TypeProto);
                    type_ptr.* = try TypeProto.parse(&type_reader);
                    value_info.type = type_ptr;
                },
                3 => { // doc_string
                    std.debug.print("\n ................ ValueInfoProto READING doc_string ", .{});
                    value_info.doc_string = try reader.readString(reader.allocator);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for ValueInfoProto", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return value_info;
    }

    pub fn print(self: *ValueInfoProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };
        std.debug.print("{s}------------- VALUEINFO \n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Name: (none)\n", .{space});
        }

        if (self.type) |t| {
            std.debug.print("{s}Type:\n", .{space});
            t.print(space);
        } else {
            std.debug.print("{s}Type: (none)\n", .{space});
        }

        if (self.doc_string) |doc| {
            std.debug.print("{s}Doc: {s}\n", .{ space, doc });
        } else {
            std.debug.print("{s}Doc: (none)\n", .{space});
        }
    }
};
