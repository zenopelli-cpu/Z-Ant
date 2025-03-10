const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const TensorShapeProto = @import("onnx.zig").TensorShapeProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

//https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L719
//TAG oneof:
//  - 1: tensor_type, type: TypeProto.Tensor
//  - 4: sequence_type, type: TypeProto.Sequence
//  - 5: map_type, type: TypeProto.Map
//  - 6: denotation, type: []const u8
//  - 8: TODO sparse_tensor_type, type: TypeProto.SparseTensor
//  - 9: TODO: optional_type, type: TypeProto.Optional
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
                        tensor.elem_type = @intCast(try reader.readVarint());
                    },
                    2 => { //shape
                        std.debug.print("\n .............................. Tensor READING shape ", .{});

                        var shape_reader = try reader.readLengthDelimited(); //var shape_reader
                        const shape_ptr = try reader.allocator.create(TensorShapeProto);
                        shape_ptr.* = try TensorShapeProto.parse(&shape_reader);
                        tensor.shape = shape_ptr;
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TensorProto\n\n", .{tag});
                        try reader.skipField(tag.wire_type);
                    },
                }
            }

            return tensor;
        }

        pub fn print(self: *Tensor, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- TENSOR_TYPE\n", .{space});

            std.debug.print("{s}Element Type: {}\n", .{ space, self.elem_type });

            if (self.shape) |s| {
                std.debug.print("{s}Shape:\n", .{space});
                s.print(space);
            } else {
                std.debug.print("{s}Shape: (none)\n", .{space});
            }
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
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for ", .{tag});
                        unreachable;
                    },
                }
            }

            return sequence;
        }

        pub fn print(self: *Sequence, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- SEQUENCE\n", .{space});

            if (self.elem_type) |t| {
                std.debug.print("{s}Element Type:\n", .{space});
                t.print(space);
            } else {
                std.debug.print("{s}Element Type: (none)\n", .{space});
            }
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

        pub fn print(self: *Map, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- MAP\n", .{space});

            std.debug.print("{s}Key Type: {}\n", .{ space, self.key_type });

            if (self.value_type) |v| {
                std.debug.print("{s}Value Type:\n", .{space});
                v.print(space);
            } else {
                std.debug.print("{s}Value Type: (none)\n", .{space});
            }
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

        pub fn print(self: *SparseTensor, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- SparseTensor\n", .{space});
            std.debug.print("{s}Element Type: {}\n", .{ space, self.elem_type });

            if (self.shape) |s| {
                std.debug.print("{s}Shape:\n", .{space});
                s.print(space);
            } else {
                std.debug.print("{s}Shape: (none)\n", .{space});
            }
        }
    };

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

        pub fn print(self: *Optional, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- OPTIONAL\n", .{space});
            if (self.elem_type) |t| {
                std.debug.print("{s}Element Type:\n", .{space});
                t.print(space);
            } else {
                std.debug.print("{s}Element Type: (none)\n", .{space});
            }
        }
    };

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
        var typeProto = TypeProto{
            .tensor_type = null,
            .sequence_type = null,
            .map_type = null,
            .sparse_tensor_type = null, //TODO
            .optional_type = null,
            .denotation = null,
        };

        _ = &typeProto;

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            std.debug.print("\n ........................ TypeProto TAG: {any} ", .{tag});

            switch (tag.field_number) {
                1 => { //tensor_type
                    std.debug.print("\n ........................ TypeProto READING tensor_type ", .{});

                    var tensor_type_reader = try reader.readLengthDelimited();
                    const ensor_type_ptr = try reader.allocator.create(Tensor);
                    ensor_type_ptr.* = try Tensor.parse(&tensor_type_reader);
                    typeProto.tensor_type = ensor_type_ptr;
                },
                4 => { //TODO sequence_type
                    std.debug.print("\n ........................ TypeProto READING sequence_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                5 => { //TODO map_type
                    std.debug.print("\n ........................ TypeProto READING map_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                6 => { // TODO denotation
                    std.debug.print("\n ........................ TypeProto READING denotation ", .{});
                    _ = try reader.readLengthDelimited();
                },
                8 => { // TODO sparse_tensor_type
                    std.debug.print("\n ........................ TypeProto READING sparse_tensor_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                9 => { // TODO optional_type
                    std.debug.print("\n ........................ TypeProto READING sparse_tensor_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TypeProto", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return typeProto;
    }

    pub fn print(self: *TypeProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- TYPE\n", .{space});

        if (self.tensor_type) |t| {
            std.debug.print("{s}Tensor Type:\n", .{space});
            t.print(space);
        } else {
            std.debug.print("{s}Tensor Type: (none)\n", .{space});
        }

        if (self.sequence_type) |s| {
            std.debug.print("{s}Sequence Type:\n", .{space});
            s.print(space);
        } else {
            std.debug.print("{s}Sequence Type: (none)\n", .{space});
        }

        if (self.map_type) |m| {
            std.debug.print("{s}Map Type:\n", .{space});
            m.print(space);
        } else {
            std.debug.print("{s}Map Type: (none)\n", .{space});
        }

        if (self.sparse_tensor_type) |st| {
            std.debug.print("{s}Sparse Tensor Type:\n", .{space});
            st.print(space);
        } else {
            std.debug.print("{s}Sparse Tensor Type: (none)\n", .{space});
        }

        if (self.optional_type) |o| {
            std.debug.print("{s}Optional Type:\n", .{space});
            o.print(space);
        } else {
            std.debug.print("{s}Optional Type: (none)\n", .{space});
        }

        if (self.denotation) |d| {
            std.debug.print("{s}Denotation: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}Denotation: (none)\n", .{space});
        }
    }
};
