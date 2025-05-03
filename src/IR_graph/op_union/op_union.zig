const std = @import("std");
const zant = @import("../../zant.zig");
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

const allocator = std.heap.page_allocator;
pub const operators = @import("operators/operators.zig");

pub const Op_union = union(enum) {
    add: operators.Add,
    averagePool: operators.AveragePool,
    batchNormalization: operators.BatchNormalization,
    ceil: operators.Ceil,
    concat: operators.Concat,
    conv: operators.Conv,
    div: operators.Div,
    elu: operators.Elu,
    flatten: operators.Flatten,
    gemm: operators.Gemm,
    reduceMean: operators.ReduceMean,

    useless: operators.Useless,

    pub fn init(nodeProto: *NodeProto) !Op_union {
        const op_type = nodeProto.op_type;

        if (std.mem.indexOf(u8, op_type, "Add")) |_| {
            return Op_union{
                .add = try operators.Add.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "AveragePool")) |_| {
            return Op_union{
                .averagePool = try operators.AveragePool.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "BatchNormalization")) |_| {
            return Op_union{
                .batchNormalization = try operators.BatchNormalization.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Ceil")) |_| {
            return Op_union{
                .ceil = try operators.Ceil.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Concat")) |_| {
            return Op_union{
                .concat = try operators.Concat.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Conv")) |_| {
            return Op_union{
                .conv = try operators.Conv.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Div")) |_| {
            return Op_union{
                .div = try operators.Div.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Elu")) |_| {
            return Op_union{
                .elu = try operators.Elu.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Flatten")) |_| {
            return Op_union{
                .flatten = try operators.Flatten.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "Gemm")) |_| {
            return Op_union{
                .gemm = try operators.Gemm.init(nodeProto),
            };
        } else if (std.mem.indexOf(u8, op_type, "ReduceMean")) |_| {
            return Op_union{
                .reduceMean = try operators.ReduceMean.init(nodeProto),
            };
        } else {
            std.debug.print("\n\nERROR: init() is not available for {s} operator!! \n\n", .{op_type});
            return Op_union{
                .useless = try operators.Useless.init(nodeProto),
            };
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
