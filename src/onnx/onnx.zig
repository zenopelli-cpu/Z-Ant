//! Stop whatever you are doing and read this before proceding!
//! https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

const std = @import("std");
const protobuf = @import("protobuf.zig");

pub const Version = enum(i64) {
    IR_VERSION_2017_10_10 = 0x0000000000000001,
    IR_VERSION_2017_10_30 = 0x0000000000000002,
    IR_VERSION_2017_11_3 = 0x0000000000000003,
    IR_VERSION_2019_1_22 = 0x0000000000000004,
    IR_VERSION_2019_3_18 = 0x0000000000000005,
    IR_VERSION_2019_9_19 = 0x0000000000000006,
    IR_VERSION_2020_5_8 = 0x0000000000000007,
    IR_VERSION_2021_7_30 = 0x0000000000000008,
    IR_VERSION_2023_5_5 = 0x0000000000000009,
    IR_VERSION_2024_3_25 = 0x000000000000000A,
    IR_VERSION = 0x000000000000000B,
};

pub const DataType = enum(i32) {
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
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
    FLOAT4E2M1 = 23,
};

pub const AttributeType = enum {
    UNDEFINED,
    FLOAT,
    INT,
    STRING,
    TENSOR,
    GRAPH,
    SPARSE_TENSOR,
    FLOATS,
    INTS,
    STRINGS,
    TENSORS,
    GRAPHS,
    SPARSE_TENSORS,
};

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

                    _ = try reader.readLengthDelimited(); //var doc_reader
                    // _ = try doc_reader.readString(reader.allocator);
                },
                else => {
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return value_info;
    }
};

pub const AttributeProto = struct {
    name: []const u8,
    type: AttributeType,
    f: f32 = 0,
    i: i64 = 0,
    s: []const u8 = "",
    t: ?*TensorProto = null,
    floats: []f32 = &[_]f32{},
    ints: []i64 = &[_]i64{},
    strings: [][]const u8 = &[_][]const u8{},

    pub fn deinit(self: *AttributeProto, allocator: std.mem.Allocator) void {
        allocator.free(self.name);

        switch (self.type) {
            .FLOAT => {},
            .INT => {},
            .STRING => allocator.free(self.s),
            .TENSOR => if (self.t) |t| {
                t.deinit(allocator);
                allocator.destroy(t);
            },
            .FLOATS => allocator.free(self.floats),
            .INTS => allocator.free(self.ints),
            .STRINGS => {
                for (self.strings) |s| allocator.free(s);
                allocator.free(self.strings);
            },
            else => {},
        }
    }

    fn parseSingleAttribute(attr_reader: *protobuf.ProtoReader, allocator: std.mem.Allocator) !AttributeProto {
        var attr = AttributeProto{
            .name = "",
            .type = .UNDEFINED,
        };

        var floats_list = std.ArrayList(f32).init(allocator);
        defer floats_list.deinit();
        var ints_list = std.ArrayList(i64).init(allocator);
        defer ints_list.deinit();
        var strings_list = std.ArrayList([]const u8).init(allocator);
        defer {
            for (strings_list.items) |s| allocator.free(s);
            strings_list.deinit();
        }

        errdefer {
            for (strings_list.items) |s| allocator.free(s);
        }

        while (attr_reader.hasMore()) {
            const attr_tag = try attr_reader.readTag();
            //DEBUG
            //std.debug.print("Parsing attribute field {d} with wire type {}\n", .{ attr_tag.field_number, attr_tag.wire_type });
            switch (attr_tag.field_number) {
                1 => { // name
                    attr.name = try attr_reader.readString(allocator);
                    // Pre-set type for known Conv attributes
                    if (std.mem.eql(u8, attr.name, "dilations") or
                        std.mem.eql(u8, attr.name, "kernel_shape") or
                        std.mem.eql(u8, attr.name, "pads") or
                        std.mem.eql(u8, attr.name, "strides"))
                    {
                        attr.type = .INTS;
                    }
                },
                20 => { // type
                    const value = try attr_reader.readVarint();
                    // Only set type if it's not already set to INTS
                    if (attr.type != .INTS) {
                        attr.type = @enumFromInt(@as(u8, @intCast(value)));
                    }
                },
                2 => { // single float (f)
                    const value = try attr_reader.readFixed32();
                    attr.f = @bitCast(value);
                    if (attr.type != .INTS) attr.type = .FLOAT;
                },
                3 => { // single int (i)
                    const value = try attr_reader.readVarint();
                    attr.i = @intCast(value);
                    if (attr.type != .INTS) attr.type = .INT;
                },
                4 => { // single string (s)
                    attr.s = try attr_reader.readString(allocator);
                    if (attr.type != .INTS) attr.type = .STRING;
                },
                5 => { // single tensor (t)
                    var tensor_reader = try attr_reader.readLengthDelimited();
                    const tensor_ptr = try allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    attr.t = tensor_ptr;
                    if (attr.type != .INTS) attr.type = .TENSOR;
                },
                6 => { // repeated float (floats)
                    if (attr_tag.wire_type == .LengthDelimited) {
                        var floats_reader = try attr_reader.readLengthDelimited();
                        while (floats_reader.hasMore()) {
                            if (floats_reader.available() < 4) break;
                            const v = try floats_reader.readFixed32();
                            try floats_list.append(@bitCast(v));
                        }
                    } else {
                        const v = try attr_reader.readFixed32();
                        try floats_list.append(@bitCast(v));
                    }
                    if (attr.type != .INTS) attr.type = .FLOATS;
                },
                7, 8 => { // repeated int64 (ints) or potential repeated int
                    const v = try attr_reader.readVarint();
                    try ints_list.append(@intCast(v));
                    //DEBUG
                    //std.debug.print("Added int value {d} to {s}\n", .{ v, attr.name });
                    if (attr.type != .INTS) attr.type = .INTS;
                },
                else => {
                    try attr_reader.skipField(attr_tag.wire_type);
                },
            }
        }

        switch (attr.type) {
            .FLOATS => attr.floats = try floats_list.toOwnedSlice(),
            .INTS => attr.ints = try ints_list.toOwnedSlice(),
            .STRINGS => attr.strings = try strings_list.toOwnedSlice(),
            else => {},
        }

        return attr;
    }
};

pub const TensorShapeProto = struct {
    dims: []i64,

    pub fn deinit(self: *TensorShapeProto, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TensorShapeProto {
        var shape = TensorShapeProto{
            .dims = &[_]i64{},
        };

        var dims_list = std.ArrayList(i64).init(reader.allocator);
        defer dims_list.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // dim
                    var dim_reader = try reader.readLengthDelimited();
                    while (dim_reader.hasMore()) {
                        const dim_tag = try dim_reader.readTag();
                        if (dim_tag.field_number == 1) { // dim_value
                            const dim_value = try dim_reader.readInt64();
                            try dims_list.append(dim_value);
                        } else {
                            try dim_reader.skipField(dim_tag.wire_type);
                        }
                    }
                },
                else => {
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        shape.dims = try dims_list.toOwnedSlice();
        return shape;
    }
};

//https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L719
//TAG oneof:
//  - 1: tensor_type, type: TypeProto.Tensor
//  - 4: sequence_type, type: TypeProto.Sequence
//  - 5: map_type, type: TypeProto.Map
//  - 6: denotation, type: []const u8
//  - 8: TODO sparse_tensor_type, type: TypeProto.SparseTensor
//  - 9: optional_type, type: TypeProto.Optional
pub const TypeProto = struct {
    //TENSOR TAG:
    //  - 1: elem_type int32
    //  - 2: shape TensorShapeProto
    pub const Tensor = struct {
        elem_type: u32,
        shape: ?*TensorShapeProto,

        pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
            if (self.shape) |s| {
                s.deinit(allocator);
                allocator.destroy(s);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Tensor {
            var tensor = Tensor{
                .elem_type = 0,
                .shape = null,
            };

            _ = &tensor;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. tensor TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Tensor READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    2 => { //shape
                        std.debug.print("\n .............................. Tensor READING tensor_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return tensor;
        }
    };

    //SEQUENCE TAG:
    //  - 1: elem_type TypeProto
    pub const Sequence = struct {
        elem_type: ?*TypeProto,

        pub fn deinit(self: *Sequence, allocator: std.mem.Allocator) void {
            if (self.elem_type) |e| {
                e.deinit(allocator);
                allocator.destroy(e);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Sequence {
            var sequence = Sequence{
                .elem_type = null,
            };

            _ = &sequence;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. Sequence TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Sequence READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return sequence;
        }
    };

    //MAP TAG:
    //  - 1: key_type u32
    //  - 2: value_type TypeProto
    pub const Map = struct {
        key_type: u32,
        value_type: ?*TypeProto,

        pub fn deinit(self: *Map, allocator: std.mem.Allocator) void {
            if (self.value_type) |v| {
                v.deinit(allocator);
                allocator.destroy(v);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Tensor {
            var map = Map{
                .key_type = 0,
                .value_type = null,
            };

            _ = &map;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. Map TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Map READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    2 => { //value_type
                        std.debug.print("\n .............................. Map READING value_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return map;
        }
    };

    //SPARSE TENSOR
    //  - 1: elem_type int32
    //  - 2: shape TensorShapeProto
    pub const SparseTensor = struct {
        elem_type: u32,
        shape: ?*TensorShapeProto,

        pub fn deinit(self: *SparseTensor, allocator: std.mem.Allocator) void {
            if (self.shape) |s| {
                s.deinit(allocator);
                allocator.destroy(s);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !SparseTensor {
            var sparse_tensor = SparseTensor{
                .elem_type = 0,
                .shape = null,
            };

            _ = &sparse_tensor;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. tensor TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Tensor READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    2 => { //shape
                        std.debug.print("\n .............................. Tensor READING tensor_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return sparse_tensor;
        }
    }; //TODO

    //TAG OPTIONAL
    //  - 1: elem_type TypeProto
    pub const Optional = struct {
        elem_type: ?*TypeProto,

        pub fn deinit(self: *Optional, allocator: std.mem.Allocator) void {
            if (self.elem_type) |e| {
                e.deinit(allocator);
                allocator.destroy(e);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Sequence {
            var opt = Optional{
                .elem_type = null,
            };

            _ = &opt;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. Optional TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Optional READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return opt;
        }
    };

    elem_type: u32,
    tensor_type: ?*Tensor,
    sequence_type: ?*Sequence,
    map_type: ?*Map,
    sparse_tensor_type: ?*SparseTensor, //TODO
    optional_type: ?*Optional,
    denotation: ?[]const u8,

    pub fn deinit(self: *TypeProto, allocator: std.mem.Allocator) void {
        if (self.tensor_type) |s| {
            s.deinit(allocator);
            allocator.destroy(s);
        }
        if (self.sequence_type) |st| {
            st.deinit(allocator);
            allocator.destroy(st);
        }
        if (self.map_type) |m| {
            m.deinit(allocator);
            allocator.destroy(m);
        }
        if (self.sparse_tensor_type) |stt| {
            stt.deinit(allocator);
            allocator.destroy(stt);
        }
        if (self.optional_type) |ot| {
            ot.deinit(allocator);
            allocator.destroy(ot);
        }

        if (self.denotation) |d| allocator.free(d);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TypeProto {
        var tensor_type = TypeProto{
            .elem_type = 0,
            .tensor_type = null,
            .sequence_type = null,
            .map_type = null,
            .sparse_tensor_type = null, //TODO
            .optional_type = null,
            .denotation = null,
        };

        _ = &tensor_type;

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            std.debug.print("\n ........................ TypeProto TAG: {any} ", .{tag});

            switch (tag.field_number) {
                1 => { //tensor_type
                    std.debug.print("\n ........................ TypeProto READING tensor_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                4 => { //sequence_type
                    std.debug.print("\n ........................ TypeProto READING sequence_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                5 => { //map_type
                    std.debug.print("\n ........................ TypeProto READING map_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                6 => { //denotation
                    std.debug.print("\n ........................ TypeProto READING denotation ", .{});
                    _ = try reader.readLengthDelimited();
                },
                8 => { // TODO sparse_tensor_type
                    std.debug.print("\n ........................ TypeProto READING sparse_tensor_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                9 => { //optional_type
                    std.debug.print("\n ........................ TypeProto READING sparse_tensor_type ", .{});
                    _ = try reader.readLengthDelimited();
                },

                else => {
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return tensor_type;
    }
};

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
                else => try reader.skipField(tag.wire_type),
            }
        }

        tensor.dims = try dims.toOwnedSlice();
        return tensor;
    }
};

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
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        node.input = try inputs.toOwnedSlice();
        node.output = try outputs.toOwnedSlice();
        node.attribute = try attributes.toOwnedSlice();
        return node;
    }
};

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L460
//TAGS:
//  - 1 : node, type: NodeProto repeated
//  - 2 : name
//  - 5 : initializer, type: TensorProto repeated
//  - 10: doc_string
//  - 11: input, type: ValueInfoProto repeated
//  - 12: output, type: ValueInfoProto repeated
//  - 13: value_info, type: ValueInfoProto repeated
//  - 14: quantization_annotation, type: TensorAnnotation repeated
//  - 15: sparse_initializer, type: TensorProto repeated
//  - 16: metadata_props, type: StringStringEntryProto repeated
//  - 3, 4, 6, 7, 8, 9 are reserved
pub const GraphProto = struct {
    name: ?[]const u8,
    nodes: []*NodeProto,
    initializers: []*TensorProto,
    inputs: []*ValueInfoProto,

    pub fn deinit(self: *GraphProto, allocator: std.mem.Allocator) void {
        if (self.name) |n| allocator.free(n);
        for (self.nodes) |node| {
            node.deinit(allocator);
            allocator.destroy(node);
        }
        allocator.free(self.nodes);
        for (self.initializers) |init| {
            init.deinit(allocator);
            allocator.destroy(init);
        }
        allocator.free(self.initializers);
        for (self.inputs) |input| { // Add this block
            input.deinit(allocator);
            allocator.destroy(input);
        }
        allocator.free(self.inputs);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !GraphProto {
        var graph = GraphProto{
            .name = null,
            .nodes = &[_]*NodeProto{},
            .initializers = &[_]*TensorProto{},
            .inputs = &[_]*ValueInfoProto{},
        };

        var nodes = std.ArrayList(*NodeProto).init(reader.allocator);
        defer nodes.deinit();
        var initializers = std.ArrayList(*TensorProto).init(reader.allocator);
        defer initializers.deinit();
        var inputs = std.ArrayList(*ValueInfoProto).init(reader.allocator);
        defer inputs.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // node
                    var node_reader = try reader.readLengthDelimited();
                    const node_ptr = try reader.allocator.create(NodeProto);
                    node_ptr.* = try NodeProto.parse(&node_reader);
                    try nodes.append(node_ptr);
                },
                2 => { // name
                    graph.name = try reader.readString(reader.allocator);
                },
                5 => { // initializer (repeated)
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    try initializers.append(tensor_ptr);
                },
                10 => { // doc_string
                    var str_reader = try reader.readLengthDelimited();
                    _ = try str_reader.readString(reader.allocator);
                },

                11 => { // input
                    std.debug.print("\n\n ........GRAPH PROTO READING input ", .{});
                    var input_reader = try reader.readLengthDelimited(); //var input_reader
                    const input_ptr = try reader.allocator.create(ValueInfoProto);
                    input_ptr.* = try ValueInfoProto.parse(&input_reader);
                    try inputs.append(input_ptr);
                },
                12 => { // output
                    // TODO: This field contains a list of ValueInfoProto messages, each representing an output of the graph.
                    // It provides information about the outputs' names, types, and shapes.
                    std.debug.print("\n\n ........GRAPH PROTO READING output ", .{});
                    _ = try reader.readLengthDelimited();
                },
                13 => { // value_info
                    //TODO: This optional field holds a list of ValueInfoProto messages that describe intermediate values within the graph.
                    //While it's not mandatory for a value to appear in this list, when present, it offers detailed information about the values computed at various stages of the graph.
                    std.debug.print("\n\n ........GRAPH PROTO READING value_info ", .{});
                    _ = try reader.readLengthDelimited();
                },
                15 => { // sparse_initializer
                    var sparse_reader = try reader.readLengthDelimited();
                    while (sparse_reader.hasMore()) {
                        _ = try sparse_reader.readVarint();
                    }
                },

                14 => { // quantization_annotation
                    //This field carries information mapping tensors to their quantization parameters, such as scale and zero-point tensors.
                    // For instance, for a tensor 'a', this field might indicate that 'a_scale' and 'a_zero_point' are its associated quantization parameters.
                    std.debug.print("\n\n ........GRAPH PROTO READING quantization_annotation ", .{});

                    _ = try reader.readLengthDelimited();
                },
                else => {
                    std.debug.print("\n\n ........default readLenghtDelimited, TAG:{any} ", .{tag});

                    var unknown_reader = try reader.readLengthDelimited();
                    while (unknown_reader.hasMore()) {
                        _ = try unknown_reader.readVarint();
                    }
                },
            }
        }

        graph.nodes = try nodes.toOwnedSlice();
        graph.initializers = try initializers.toOwnedSlice();
        graph.inputs = try inputs.toOwnedSlice();
        return graph;
    }
};

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
                else => try reader.skipField(tag.wire_type),
            }
        }

        return model;
    }
};

pub fn parseFromFile(allocator: std.mem.Allocator, file_path: []const u8) !ModelProto {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, @intCast(file_size));
    defer allocator.free(buffer);

    const bytes_read = try file.readAll(buffer);
    if (bytes_read != file_size) {
        return error.UnexpectedEOF;
    }

    var reader = protobuf.ProtoReader.init(allocator, buffer);
    var model = try ModelProto.parse(&reader);
    errdefer model.deinit(allocator);

    return model;
}

fn printTensorData(data: []const u8, data_type: DataType) void {
    switch (data_type) {
        .FLOAT => {
            const float_slice = @as([*]const f32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            const num_to_print = @min(float_slice.len, 10);
            for (float_slice[0..num_to_print]) |val| {
                std.debug.print("{d:.3} ", .{val});
            }
            if (float_slice.len > 10) {
                std.debug.print("...", .{});
            }
        },
        .INT32 => {
            const int_slice = @as([*]const i32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            const num_to_print = @min(int_slice.len, 10);
            for (int_slice[0..num_to_print]) |val| {
                std.debug.print("{d} ", .{val});
            }
            if (int_slice.len > 10) {
                std.debug.print("...", .{});
            }
        },
        .INT64 => {
            const int_slice = @as([*]const i64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            const num_to_print = @min(int_slice.len, 10);
            for (int_slice[0..num_to_print]) |val| {
                std.debug.print("{d} ", .{val});
            }
            if (int_slice.len > 10) {
                std.debug.print("...", .{});
            }
        },
        else => {
            std.debug.print("(data type {s} not supported for display)", .{@tagName(data_type)});
        },
    }
}

pub fn printStructure(model: *ModelProto) void {
    // Print model info
    std.debug.print("\n=== Model Info ===\n", .{});
    std.debug.print("IR Version: {}\n", .{model.ir_version});
    if (model.producer_name) |name| {
        std.debug.print("Producer: {s}\n", .{name});
    }
    if (model.producer_version) |version| {
        std.debug.print("Version: {s}\n", .{version});
    }

    // Print graph info
    if (model.graph) |graph| {
        std.debug.print("\n=== Graph Info ===\n", .{});
        if (graph.name) |name| {
            std.debug.print("Name: {s}\n", .{name});
        }
        std.debug.print("Nodes: {d}\n", .{graph.nodes.len});
        std.debug.print("Initializers: {d}\n", .{graph.initializers.len});
        std.debug.print("Inputs: {d}\n", .{graph.inputs.len});

        // First, print a high-level view of the graph structure
        std.debug.print("\n=== Graph Structure ===\n", .{});
        for (graph.nodes, 0..) |node_ptr, i| {
            // Print current node
            std.debug.print("\n[{d}] {s}", .{ i, node_ptr.op_type });
            if (node_ptr.name) |name| {
                std.debug.print(" ({s})", .{name});
            }
            std.debug.print("\n", .{});

            // Print inputs with arrows
            std.debug.print("  Inputs:\n", .{});
            for (node_ptr.input) |input| {
                std.debug.print("    ← {s}\n", .{input});
            }

            // Print outputs with arrows
            std.debug.print("  Outputs:\n", .{});
            for (node_ptr.output) |output| {
                std.debug.print("    → {s}\n", .{output});
            }
        }

        // Then print detailed node information
        std.debug.print("\n=== Detailed Node Info ===\n", .{});
        for (graph.nodes, 0..) |node_ptr, i| {
            std.debug.print("\n[Node {d}]\n", .{i});
            if (node_ptr.name) |name| {
                std.debug.print("Name: {s}\n", .{name});
            }
            std.debug.print("Type: {s}\n", .{node_ptr.op_type});
            if (node_ptr.domain) |domain| {
                std.debug.print("Domain: {s}\n", .{domain});
            }

            // Print attributes
            if (node_ptr.attribute.len > 0) {
                std.debug.print("Attributes:\n", .{});
                for (node_ptr.attribute) |attr| {
                    std.debug.print("  {s}: ", .{attr.name});
                    switch (attr.type) {
                        .FLOAT => std.debug.print("float = {d}\n", .{attr.f}),
                        .INT => std.debug.print("int = {d}\n", .{attr.i}),
                        .STRING => std.debug.print("string = {s}\n", .{attr.s}),
                        .TENSOR => {
                            std.debug.print("tensor = ", .{});
                            if (attr.t) |t| {
                                std.debug.print("type: {}, shape: [", .{t.data_type});
                                for (t.dims, 0..) |dim, j| {
                                    if (j > 0) std.debug.print(", ", .{});
                                    std.debug.print("{d}", .{dim});
                                }
                                std.debug.print("]\n", .{});

                                // Print tensor data if available
                                if (t.float_data) |data| {
                                    std.debug.print("    data = [", .{});
                                    for (data[0..@min(data.len, 10)]) |val| {
                                        std.debug.print("{d:.3} ", .{val});
                                    }
                                    if (data.len > 10) {
                                        std.debug.print("...", .{});
                                    }
                                    std.debug.print("]\n", .{});
                                } else if (t.raw_data) |data| {
                                    std.debug.print("    raw_data = [", .{});
                                    printTensorData(data, t.data_type);
                                    std.debug.print("]\n", .{});
                                } else if (t.int32_data) |data| {
                                    std.debug.print("    int32_data = [", .{});
                                    for (data[0..@min(data.len, 10)]) |val| {
                                        std.debug.print("{d} ", .{val});
                                    }
                                    if (data.len > 10) {
                                        std.debug.print("...", .{});
                                    }
                                    std.debug.print("]\n", .{});
                                } else if (t.int64_data) |data| {
                                    std.debug.print("    int64_data = [", .{});
                                    for (data[0..@min(data.len, 10)]) |val| {
                                        std.debug.print("{d} ", .{val});
                                    }
                                    if (data.len > 10) {
                                        std.debug.print("...", .{});
                                    }
                                    std.debug.print("]\n", .{});
                                } else {
                                    std.debug.print("    (no data available)\n", .{});
                                }
                            } else {
                                std.debug.print("null\n", .{});
                            }
                        },
                        .FLOATS => {
                            std.debug.print("floats = [", .{});
                            for (attr.floats) |f| {
                                std.debug.print("{d} ", .{f});
                            }
                            std.debug.print("]\n", .{});
                        },
                        .INTS => {
                            std.debug.print("ints = [", .{});
                            for (attr.ints) |val| {
                                std.debug.print("{d} ", .{val});
                            }
                            std.debug.print("]\n", .{});
                        },
                        .STRINGS => {
                            std.debug.print("strings = [", .{});
                            for (attr.strings) |s| {
                                std.debug.print("{s} ", .{s});
                            }
                            std.debug.print("]\n", .{});
                        },
                        else => std.debug.print("unsupported type\n", .{}),
                    }
                }
            }
        }

        // Print initializer details
        std.debug.print("\n=== Initializers (weights, biases, etc.) ===\n", .{});
        for (graph.initializers, 0..) |init_ptr, i| {
            std.debug.print("\nInitializer {d}:\n", .{i});
            if (init_ptr.name) |name| {
                if (std.mem.indexOf(u8, name, "weight")) |_| {
                    std.debug.print("Name: {s} (weights/filters)\n", .{name});
                } else if (std.mem.indexOf(u8, name, "bias")) |_| {
                    std.debug.print("Name: {s} (bias values)\n", .{name});
                } else if (std.mem.indexOf(u8, name, "running_mean")) |_| {
                    std.debug.print("Name: {s} (batch norm mean)\n", .{name});
                } else if (std.mem.indexOf(u8, name, "running_var")) |_| {
                    std.debug.print("Name: {s} (batch norm variance)\n", .{name});
                } else {
                    std.debug.print("Name: {s}\n", .{name});
                }
            }
            std.debug.print("Type: {}\n", .{init_ptr.data_type});
            std.debug.print("Shape: [", .{});
            for (init_ptr.dims, 0..) |dim, j| {
                if (j > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("]\n", .{});

            // Print some data samples
            std.debug.print("Data preview: ", .{});
            if (init_ptr.float_data) |data| {
                std.debug.print(" float_data [", .{});
                for (data[0..@min(data.len, 5)]) |val| {
                    std.debug.print("{d:.3} ", .{val});
                }
                if (data.len > 5) {
                    std.debug.print("...", .{});
                }
                std.debug.print("]\n", .{});
            } else if (init_ptr.raw_data) |data| {
                std.debug.print(" raw_data [", .{});
                printTensorData(data, init_ptr.data_type);
                std.debug.print("]\n", .{});
            } else if (init_ptr.int32_data) |data| {
                std.debug.print(" int32_data [", .{});
                for (data[0..@min(data.len, 5)]) |val| {
                    std.debug.print("{d} ", .{val});
                }
                if (data.len > 5) {
                    std.debug.print("...", .{});
                }
                std.debug.print("]\n", .{});
            } else {
                std.debug.print("(no data available)\n", .{});
            }

            // Add explanation based on shape
            if (init_ptr.dims.len > 0) {
                std.debug.print("Description: ", .{});
                switch (init_ptr.dims.len) {
                    1 => std.debug.print("1D tensor with {d} values (typically bias or batch norm parameter)\n", .{init_ptr.dims[0]}),
                    2 => std.debug.print("2D matrix of size {d}x{d} (typically dense layer weights)\n", .{ init_ptr.dims[0], init_ptr.dims[1] }),
                    4 => std.debug.print("4D tensor of size {d}x{d}x{d}x{d} (convolutional filters: out_channels x in_channels x height x width)\n", .{ init_ptr.dims[0], init_ptr.dims[1], init_ptr.dims[2], init_ptr.dims[3] }),
                    else => std.debug.print("{d}D tensor\n", .{init_ptr.dims.len}),
                }
            }
        }
    }
}
