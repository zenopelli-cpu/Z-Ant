const std = @import("std");
const zant = @import("../../zant.zig");
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

const allocator = std.heap.page_allocator;
pub const operators = @import("operators/operators.zig");

pub const Op_union = union(enum) {
    add: operators.Add,
    sub: operators.Sub,
    conv: operators.Conv,
    useless: operators.Useless,

    pub fn init(nodeProto: *NodeProto) !Op_union {
        const op_type = nodeProto.op_type;

        if (std.mem.indexOf(u8, op_type, "Add")) |_| {
            return Op_union{
                .add = try operators.Add.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Sub")) |_| {
            return Op_union{
                .sub = try operators.Sub.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Conv")) |_| {
            return Op_union{
                .conv = try operators.Conv.init(nodeProto),
            };
        } else {
            std.debug.print("\n\nERROR: init() is not available for {s} operator!! \n\n", .{op_type});
            //return error.OpNotAvailable;
        }

        return Op_union{
            .useless = try operators.Useless.init(nodeProto),
        };
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
