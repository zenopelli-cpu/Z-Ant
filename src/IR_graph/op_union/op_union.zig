const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

const allocator = std.heap.page_allocator;
const op = @import("operators/operators.zig");

pub const Op_union = union(enum) {
    add: op.Add,
    sub: op.Sub,
    conv: op.Conv,

    pub fn init(nodeProto: NodeProto) !Op_union {
        const op_type = nodeProto.op_type;

        if (std.mem.indexOf(u8, op_type, "Add")) |_| {
            return Op_union{
                .add = op.Add.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Sub")) |_| {
            return Op_union{
                .sub = op.Sub.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Conv")) |_| {
            return Op_union{
                .conv = op.Conv.init(nodeProto),
            };
        } else {
            std.debug.print("\n\nERROR: init() is not available for {s} operator!! \n\n", .{op_type});
            return error.OpNotAvailable;
        }
    }

    pub fn get_output_shape(self: Op_union) []usize {
        switch (self) {
            .add => |ptr| return ptr.get_output_shape(),
            .sub => |ptr| return ptr.get_output_shape(),
            .conv => |ptr| return ptr.get_output_shape(),
            else => {
                std.debug.print("\n\nERROR: get_output_shape() is not available!! \n\n", .{});
                return error.OpNotAvailable;
            },
        }
    }

    pub fn print(self: Op_union) void {
        switch (self) {
            .add => |ptr| ptr.print(),
            .sub => |ptr| ptr.print(),
            .conv => |ptr| ptr.print(),
            else => {
                std.debug.print("\n\nERROR: print() is not available!! \n\n", .{});
                return error.OpNotAvailable;
            },
        }
    }
};
