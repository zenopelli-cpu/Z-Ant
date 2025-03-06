const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const DataType = @import("onnx.zig").DataType;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L503
//TAGS:
//  - 1 : dims, repeated int64
//  - 2 : data_type, optional int32
//  - 3 : TODO NOT URGENT segment, optional Segment
//  - 4 : float_data, repeated float
//  - 5 : int32_data, repeated int32
//  - 6 : string_data, repeated bytes
//  - 7 : int64_data, repeated int64
//  - 8 : name, optional string
//  - 9 : raw_data, optional bytes
//  - 10: double_data, repeated double
//  - 11: uint64_data, repeated uint64
//  - 12: TODO doc_string, optional string
//  - 13: TODO external_data, repeated StringStringEntryProto
//  - 14: TODO data_location, optional DataLocation
//  - 16: TODO metadata_props, repeated StringStringEntryProto
pub const TensorProto = struct {
    dims: []i64,
    data_type: DataType,
    name: ?[]const u8,
    raw_data: ?[]const u8,
    float_data: ?[]f32,
    int32_data: ?[]i32,
    string_data: ?[][]const u8,
    int64_data: ?[]i64,
    double_data: ?[]f64,
    uint64_data: ?[]u64,

    pub fn deinit(self: *TensorProto, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
        if (self.raw_data) |data| allocator.free(data);
        if (self.float_data) |data| allocator.free(data);
        if (self.int32_data) |data| allocator.free(data);
        if (self.int64_data) |data| allocator.free(data);
        if (self.double_data) |data| allocator.free(data);
        if (self.uint64_data) |data| allocator.free(data);
        if (self.string_data) |data| {
            for (data) |str| allocator.free(str);
            allocator.free(data);
        }
        if (self.name) |n| allocator.free(n);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TensorProto {
        var tensor = TensorProto{
            .dims = &[_]i64{},
            .data_type = .UNDEFINED,
            .name = null,
            .raw_data = null,
            .float_data = null,
            .int32_data = null,
            .string_data = null,
            .int64_data = null,
            .double_data = null,
            .uint64_data = null,
        };

        var dims = std.ArrayList(i64).init(reader.allocator);
        defer dims.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // dims
                    const value = try reader.readVarint();
                    try dims.append(@as(i64, @intCast(value)));
                },
                2 => { // data_type
                    const value = try reader.readVarint();
                    tensor.data_type = @enumFromInt((value));
                },
                8 => { // name
                    tensor.name = try reader.readString(reader.allocator);
                },
                9 => { // raw_data
                    tensor.raw_data = try reader.readBytes(reader.allocator);
                },
                4 => { // float_data
                    var data = std.ArrayList(f32).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var float_reader = try reader.readLengthDelimited();
                        while (float_reader.hasMore()) {
                            if (float_reader.available() < 4) break;
                            const value = try float_reader.readFixed32();
                            try data.append(@bitCast(value));
                        }
                    } else {
                        const value = try reader.readFixed32();
                        try data.append(@bitCast(value));
                    }
                    tensor.float_data = try data.toOwnedSlice();
                },
                5 => { // int32_data
                    var data = std.ArrayList(i32).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var int_reader = try reader.readLengthDelimited();
                        while (int_reader.hasMore()) {
                            const value = try int_reader.readVarint();
                            try data.append(@intCast(value));
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(@intCast(value));
                    }
                    tensor.int32_data = try data.toOwnedSlice();
                },
                7 => { // int64_data
                    var data = std.ArrayList(i64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var int_reader = try reader.readLengthDelimited();
                        while (int_reader.hasMore()) {
                            const value = try int_reader.readVarint();
                            try data.append(@intCast(value));
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(@intCast(value));
                    }
                    tensor.int64_data = try data.toOwnedSlice();
                },
                10 => { // double_data
                    var data = std.ArrayList(f64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var double_reader = try reader.readLengthDelimited();
                        while (double_reader.hasMore()) {
                            if (double_reader.available() < 8) break;
                            const value = try double_reader.readFixed64();
                            try data.append(@bitCast(value));
                        }
                    } else {
                        const value = try reader.readFixed64();
                        try data.append(@bitCast(value));
                    }
                    tensor.double_data = try data.toOwnedSlice();
                },
                11 => { // uint64_data
                    var data = std.ArrayList(u64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var uint_reader = try reader.readLengthDelimited();
                        while (uint_reader.hasMore()) {
                            const value = try uint_reader.readVarint();
                            try data.append(value);
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(value);
                    }
                    tensor.uint64_data = try data.toOwnedSlice();
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TensorProto\n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        tensor.dims = try dims.toOwnedSlice();
        return tensor;
    }

    pub fn print(self: *TensorProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- TENSOR\n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Name: (none)\n", .{space});
        }

        std.debug.print("{s}Data Type: {any}\n", .{ space, self.data_type });

        std.debug.print("{s}Dims: [", .{space});
        for (self.dims, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{}", .{dim});
        }
        std.debug.print("]\n", .{});

        if (self.raw_data) |raw| {
            std.debug.print("{s}Raw Data: {} bytes\n", .{ space, raw.len });
        }

        if (self.float_data) |data| {
            std.debug.print("{s}Float Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.int32_data) |data| {
            std.debug.print("{s}Int32 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.int64_data) |data| {
            std.debug.print("{s}Int64 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.double_data) |data| {
            std.debug.print("{s}Double Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.uint64_data) |data| {
            std.debug.print("{s}UInt64 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.string_data) |data| {
            std.debug.print("  String Data: [", .{});
            for (data, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("\"{s}\"", .{val});
            }
            std.debug.print("]\n", .{});
        }
    }
};
