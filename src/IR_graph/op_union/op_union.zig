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
    gather: operators.Gather,
    gemm: operators.Gemm,
    identity: operators.Identity,
    leakyRelu: operators.LeakyRelu,
    matMul: operators.MatMul,
    maxPool: operators.MaxPool,
    mul: operators.Mul,
    neg: operators.Neg,
    reduceMean: operators.ReduceMean,
    relu: operators.Relu,
    reshape: operators.Reshape,
    resize: operators.Resize,
    // shape: operators.Shape,
    sigmoid: operators.Sigmoid,
    slice: operators.Slice,
    softmax: operators.Softmax,
    split: operators.Split,
    sub: operators.Sub,
    tanh: operators.Tanh,
    transpose: operators.Transpose,
    unsqueeze: operators.Unsqueeze,

    useless: operators.Useless,

    pub fn init(nodeProto: *NodeProto) !Op_union {
        const op_type = nodeProto.op_type;

        if (std.mem.eql(u8, op_type, "Add")) {
            return Op_union{ .add = try operators.Add.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "AveragePool")) {
            return Op_union{ .averagePool = try operators.AveragePool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "BatchNormalization")) {
            return Op_union{ .batchNormalization = try operators.BatchNormalization.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Ceil")) {
            return Op_union{ .ceil = try operators.Ceil.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Concat")) {
            return Op_union{ .concat = try operators.Concat.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Conv")) {
            return Op_union{ .conv = try operators.Conv.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Div")) {
            return Op_union{ .div = try operators.Div.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Elu")) {
            return Op_union{ .elu = try operators.Elu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Flatten")) {
            return Op_union{ .flatten = try operators.Flatten.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Gather")) {
            return Op_union{ .gather = try operators.Gather.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Gemm")) {
            return Op_union{ .gemm = try operators.Gemm.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Identity")) {
            return Op_union{ .identity = try operators.Identity.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "LeakyRelu")) {
            return Op_union{ .leakyRelu = try operators.LeakyRelu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "MatMul")) {
            return Op_union{ .matMul = try operators.MatMul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "MaxPool")) {
            return Op_union{ .maxPool = try operators.MaxPool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Mul")) {
            return Op_union{ .mul = try operators.Mul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Neg")) {
            return Op_union{ .neg = try operators.Neg.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "ReduceMean")) {
            return Op_union{ .reduceMean = try operators.ReduceMean.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Relu")) {
            return Op_union{ .relu = try operators.Relu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Reshape")) {
            return Op_union{ .reshape = try operators.Reshape.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Resize")) {
            return Op_union{ .resize = try operators.Resize.init(nodeProto) };
        }
        // else if (std.mem.eql(u8, op_type, "Shape")) {
        //     return Op_union{ .shape = try operators.Shape.init(nodeProto) };
        // }
        else if (std.mem.eql(u8, op_type, "Sigmoid")) {
            return Op_union{ .sigmoid = try operators.Sigmoid.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Slice")) {
            return Op_union{ .slice = try operators.Slice.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Softmax")) {
            return Op_union{ .softmax = try operators.Softmax.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Split")) {
            return Op_union{ .split = try operators.Split.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Sub")) {
            return Op_union{ .sub = try operators.Sub.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Tanh")) {
            return Op_union{ .tanh = try operators.Tanh.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Transpose")) {
            return Op_union{ .transpose = try operators.Transpose.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Unsqueeze")) {
            return Op_union{ .unsqueeze = try operators.Unsqueeze.init(nodeProto) };
        } else {
            std.debug.print("\n\nERROR: init() is not available for {s} operator!! \n\n", .{op_type});
            return Op_union{ .useless = try operators.Useless.init(nodeProto) };
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
