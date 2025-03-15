const std = @import("std");
const protobuf = @import("protobuf.zig");
const Version = @import("onnx.zig").Version;
const GraphProto = @import("onnx.zig").GraphProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L361
// TAGS:
//  - 1 : ir_version, optional int64
//  - 2 : optional string producer_name, optional string
//  - 3 : producer_version, optional string
//  - 4 : optional string domain
//  - 5 : model_version, optional int64
//  - 6 : doc_string, optional string
//  - 7 : graph, optional GraphProto
//  - 8 : TODO opset_import, repeated OperatorSetIdProto
//  - 14: TODO metadata_props, repeated StringStringEntryProto
//  - 20: TODO NOT IMPORTANT NOT URGENT training_info, repeated TrainingInfoProto
//  - 25: TODO NOT URGENT functions, repeated FunctionProto
pub const ModelProto = struct {
    ir_version: Version,
    producer_name: ?[]const u8,
    producer_version: ?[]const u8,
    domain: ?[]const u8,
    model_version: ?i64,
    doc_string: ?[]const u8,
    graph: ?*GraphProto,

    pub fn deinit(self: *ModelProto, allocator: std.mem.Allocator) void {
        if (self.producer_name) |n| allocator.free(n);
        if (self.producer_version) |v| allocator.free(v);
        if (self.domain) |d| allocator.free(d);
        if (self.doc_string) |d| allocator.free(d);
        if (self.graph) |g| {
            g.deinit(allocator);
            allocator.destroy(g);
        }
    }

    pub fn parse(reader: *protobuf.ProtoReader) !ModelProto {
        var model = ModelProto{
            .ir_version = undefined,
            .producer_name = null,
            .producer_version = null,
            .domain = null,
            .model_version = null,
            .doc_string = null,
            .graph = null,
        };
        errdefer {
            if (model.producer_name) |n| reader.allocator.free(n);
            if (model.producer_version) |v| reader.allocator.free(v);
            if (model.domain) |d| reader.allocator.free(d);
            if (model.doc_string) |d| reader.allocator.free(d);
            if (model.graph) |g| g.deinit(reader.allocator);
        }

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // ir_version
                    const value = try reader.readVarint();
                    model.ir_version = @enumFromInt(value);
                },
                2 => { // producer_name
                    const str = try reader.readString(reader.allocator);
                    if (model.producer_name) |old| reader.allocator.free(old);
                    model.producer_name = str;
                },
                3 => { // producer_version
                    const str = try reader.readString(reader.allocator);
                    if (model.producer_version) |old| reader.allocator.free(old);
                    model.producer_version = str;
                },
                4 => { // domain
                    const str = try reader.readString(reader.allocator);
                    if (model.domain) |old| reader.allocator.free(old);
                    model.domain = str;
                },
                5 => { // model_version
                    const value = try reader.readVarint();
                    model.model_version = @as(i64, @intCast(value));
                },
                6 => { // doc_string
                    const str = try reader.readString(reader.allocator);
                    if (model.doc_string) |old| reader.allocator.free(old);
                    model.doc_string = str;
                },
                7 => { // graph
                    if (model.graph) |g| {
                        g.deinit(reader.allocator);
                        reader.allocator.destroy(g);
                    }
                    var graph_reader = try reader.readLengthDelimited();
                    const graph_ptr = try reader.allocator.create(GraphProto);
                    graph_ptr.* = try GraphProto.parse(&graph_reader);
                    model.graph = graph_ptr;
                },
                else => {
                    std.debug.print("\n\n ........default readLenghtDelimited, TAG:{any} \n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return model;
    }

    pub fn print(self: *ModelProto) void {
        std.debug.print("\n\n------------------------- MODEL -------------------------------\n", .{});

        std.debug.print("ModelProto:\n", .{});
        std.debug.print("  IR Version: {}\n", .{self.ir_version});

        if (self.producer_name) |name| {
            std.debug.print("  Producer Name: {s}\n", .{name});
        } else {
            std.debug.print("  Producer Name: (none)\n", .{});
        }

        if (self.producer_version) |version| {
            std.debug.print("  Producer Version: {s}\n", .{version});
        } else {
            std.debug.print("  Producer Version: (none)\n", .{});
        }

        if (self.domain) |d| {
            std.debug.print("  Domain: {s}\n", .{d});
        } else {
            std.debug.print("  Domain: (none)\n", .{});
        }

        if (self.model_version) |v| {
            std.debug.print("  Model Version: {}\n", .{v});
        } else {
            std.debug.print("  Model Version: (none)\n", .{});
        }

        if (self.doc_string) |doc| {
            std.debug.print("  Doc String: {s}\n", .{doc});
        } else {
            std.debug.print("  Doc String: (none)\n", .{});
        }

        if (self.graph) |g| {
            std.debug.print("  Graph:\n", .{});
            g.print(null);
        } else {
            std.debug.print("  Graph: (none)\n", .{});
        }

        printingAllocator.deinit();
    }
};
