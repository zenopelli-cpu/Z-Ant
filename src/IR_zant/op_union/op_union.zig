const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

const allocator = std.heap.page_allocator;
pub const operators = @import("operators/operators.zig");

const tensorZant = @import("../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;

// --- uops ---
const UOpBuilder = zant.uops.UOpBuilder;

pub const Op_union = union(enum) {
    add: operators.Add,
    averagePool: operators.AveragePool,
    batchNormalization: operators.BatchNormalization,
    cast: operators.Cast,
    ceil: operators.Ceil,
    clip: operators.Clip,
    concat: operators.Concat,
    constant: operators.Constant,
    conv: operators.Conv,
    convClip: operators.ConvClip,
    convInteger: operators.ConvInteger,

    dequantizeLinear: operators.DequantizeLinear,
    div: operators.Div,
    dynamicQuantizeLinear: operators.DynamicQuantizeLinear,
    elu: operators.Elu,
    exp: operators.Exp,
    flatten: operators.Flatten,
    floor: operators.Floor,
    gather: operators.Gather,
    gatherND: operators.GatherND,
    gemm: operators.Gemm,
    gelu: operators.Gelu,
    globalAveragePool: operators.GlobalAveragePool,
    identity: operators.Identity,
    leakyRelu: operators.LeakyRelu,
    matMul: operators.MatMul,
    maxPool: operators.MaxPool,
    min: operators.Min,
    mul: operators.Mul,
    neg: operators.Neg,
    nonMaxSuppression: operators.NonMaxSuppression,
    oneHot: operators.OneHot,
    pad: operators.Pad,
    qgemm: operators.QGemm,
    qlinearadd: operators.QLinearAdd,
    qlinearaveragepool: operators.QLinearAveragePool,
    qlinearconcat: operators.QLinearConcat,
    qlinearconv: operators.QLinearConv,
    qlinearglobalaveragepool: operators.QLinearGlobalAveragePool,
    qlinearmatmul: operators.QLinearMatMul,
    qlinearmul: operators.QLinearMul,
    qlinearsoftmax: operators.QLinearSoftmax,
    quantizeLinear: operators.QuantizeLinear,
    reduceMean: operators.ReduceMean,
    relu: operators.Relu,
    reshape: operators.Reshape,
    resize: operators.Resize,
    shape: operators.Shape,
    sigmoid: operators.Sigmoid,
    slice: operators.Slice,
    softmax: operators.Softmax,
    split: operators.Split,
    squeeze: operators.Squeeze,
    sqrt: operators.Sqrt,
    sub: operators.Sub,
    topK: operators.TopK,
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
        } else if (std.mem.eql(u8, op_type, "Cast")) {
            return Op_union{ .cast = try operators.Cast.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Ceil")) {
            return Op_union{ .ceil = try operators.Ceil.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Clip")) {
            return Op_union{ .clip = try operators.Clip.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Concat")) {
            return Op_union{ .concat = try operators.Concat.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Constant")) {
            return Op_union{ .constant = try operators.Constant.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Conv")) {
            return Op_union{ .conv = try operators.Conv.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "ConvInteger")) {
            return Op_union{ .convInteger = try operators.ConvInteger.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "DequantizeLinear")) {
            return Op_union{ .dequantizeLinear = try operators.DequantizeLinear.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Div")) {
            return Op_union{ .div = try operators.Div.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "DynamicQuantizeLinear")) {
            return Op_union{ .dynamicQuantizeLinear = try operators.DynamicQuantizeLinear.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Elu")) {
            return Op_union{ .elu = try operators.Elu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Exp")) {
            return Op_union{ .exp = try operators.Exp.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Flatten")) {
            return Op_union{ .flatten = try operators.Flatten.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Floor")) {
            return Op_union{ .floor = try operators.Floor.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Gather")) {
            return Op_union{ .gather = try operators.Gather.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "GatherND")) {
            return Op_union{ .gatherND = try operators.GatherND.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Gelu")) {
            return Op_union{ .gelu = try operators.Gelu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Gemm")) {
            return Op_union{ .gemm = try operators.Gemm.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "GlobalAveragePool")) {
            return Op_union{ .globalAveragePool = try operators.GlobalAveragePool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Identity")) {
            return Op_union{ .identity = try operators.Identity.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "LeakyRelu")) {
            return Op_union{ .leakyRelu = try operators.LeakyRelu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "MatMul")) {
            return Op_union{ .matMul = try operators.MatMul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "MaxPool")) {
            return Op_union{ .maxPool = try operators.MaxPool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Min")) {
            return Op_union{ .min = try operators.Min.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Mul")) {
            return Op_union{ .mul = try operators.Mul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Neg")) {
            return Op_union{ .neg = try operators.Neg.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "NonMaxSuppression")) {
            return Op_union{ .nonMaxSuppression = try operators.NonMaxSuppression.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "OneHot")) {
            return Op_union{ .oneHot = try operators.OneHot.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Pad")) {
            return Op_union{ .pad = try operators.Pad.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QGemm")) {
            return Op_union{ .qgemm = try operators.QGemm.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearAdd")) {
            return Op_union{ .qlinearadd = try operators.QLinearAdd.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearAveragePool")) {
            return Op_union{ .qlinearaveragepool = try operators.QLinearAveragePool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearConcat")) {
            return Op_union{ .qlinearconcat = try operators.QLinearConcat.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearConv")) {
            return Op_union{ .qlinearconv = try operators.QLinearConv.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearGlobalAveragePool")) {
            return Op_union{ .qlinearglobalaveragepool = try operators.QLinearGlobalAveragePool.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearMatMul")) {
            return Op_union{ .qlinearmatmul = try operators.QLinearMatMul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearMul")) {
            return Op_union{ .qlinearmul = try operators.QLinearMul.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QLinearSoftmax")) {
            return Op_union{ .qlinearsoftmax = try operators.QLinearSoftmax.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "QuantizeLinear")) {
            return Op_union{ .quantizeLinear = try operators.QuantizeLinear.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "ReduceMean")) {
            return Op_union{ .reduceMean = try operators.ReduceMean.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Relu")) {
            return Op_union{ .relu = try operators.Relu.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Reshape")) {
            return Op_union{ .reshape = try operators.Reshape.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Resize")) {
            return Op_union{ .resize = try operators.Resize.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Shape")) {
            return Op_union{ .shape = try operators.Shape.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Sigmoid")) {
            return Op_union{ .sigmoid = try operators.Sigmoid.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Slice")) {
            return Op_union{ .slice = try operators.Slice.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Softmax")) {
            return Op_union{ .softmax = try operators.Softmax.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Split")) {
            return Op_union{ .split = try operators.Split.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Squeeze")) {
            return Op_union{ .squeeze = try operators.Squeeze.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Sqrt")) {
            return Op_union{ .sqrt = try operators.Sqrt.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "Sub")) {
            return Op_union{ .sub = try operators.Sub.init(nodeProto) };
        } else if (std.mem.eql(u8, op_type, "TopK")) {
            return Op_union{ .topK = try operators.TopK.init(nodeProto) };
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

    pub fn get_output_shape(self: Op_union) ![]usize {
        switch (self) {
            .add => |ptr| return ptr.get_output_shape(),
            .averagePool => |ptr| return ptr.get_output_shape(),
            .batchNormalization => |ptr| return ptr.get_output_shape(),
            .cast => |ptr| return ptr.get_output_shape(),
            .ceil => |ptr| return ptr.get_output_shape(),
            .clip => |ptr| return ptr.compute_output_shape(),
            .concat => |ptr| return ptr.get_output_shape(),
            .constant => |ptr| return ptr.get_output_shape(),
            .conv => |ptr| return ptr.get_output_shape(),
            .convClip => |ptr| return ptr.get_output_shape(),
            .convInteger => |ptr| return ptr.get_output_shape(),

            .dequantizeLinear => |ptr| return ptr.get_output_shape(),
            .div => |ptr| return ptr.get_output_shape(),
            .dynamicQuantizeLinear => |ptr| return ptr.get_output_shape(),
            .elu => |ptr| return ptr.get_output_shape(),
            .exp => |ptr| return ptr.get_output_shape(),
            .flatten => |ptr| return ptr.get_output_shape(),
            .floor => |ptr| return ptr.get_output_shape(),
            .gather => |ptr| return ptr.get_output_shape(),
            .gatherND => |ptr| return ptr.get_output_shape(),
            .gemm => |ptr| return ptr.get_output_shape(),
            .gelu => |ptr| return ptr.get_output_shape(),
            .globalAveragePool => |ptr| return ptr.get_output_shape(),
            .identity => |ptr| return ptr.get_output_shape(),
            .leakyRelu => |ptr| return ptr.get_output_shape(),
            .matMul => |ptr| return ptr.get_output_shape(),
            .maxPool => |ptr| return ptr.get_output_shape(),
            .min => |ptr| return ptr.get_output_shape(),
            .mul => |ptr| return ptr.get_output_shape(),
            .neg => |ptr| return ptr.get_output_shape(),
            .nonMaxSuppression => |ptr| return ptr.get_output_shape(),
            .oneHot => |ptr| return ptr.get_output_shape(),
            .pad => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qgemm => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearadd => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearaveragepool => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearconcat => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearconv => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearglobalaveragepool => |ptr| return ptr.compute_output_shape() catch ptr.get_output_shape(),
            .qlinearmatmul => |ptr| return ptr.get_output_shape(),
            .qlinearmul => |ptr| return ptr.get_output_shape(),
            .qlinearsoftmax => |ptr| return ptr.compute_output_shape(),
            .quantizeLinear => |ptr| return ptr.get_output_shape(),
            .reduceMean => |ptr| return ptr.get_output_shape(),
            .relu => |ptr| return ptr.get_output_shape(),
            .reshape => |ptr| return ptr.get_output_shape(),
            .resize => |ptr| return ptr.get_output_shape(),
            .shape => |ptr| return ptr.get_output_shape(),
            .sigmoid => |ptr| return ptr.get_output_shape(),
            .slice => |ptr| return ptr.get_output_shape(),
            .softmax => |ptr| return ptr.get_output_shape(),
            .split => |ptr| return ptr.get_output_shape(),
            .squeeze => |ptr| return ptr.get_output_shape(),
            .sqrt => |ptr| return ptr.get_output_shape(),
            .sub => |ptr| return ptr.get_output_shape(),
            .topK => |ptr| return ptr.get_output_shape(),
            .tanh => |ptr| return ptr.get_output_shape(),
            .transpose => |ptr| return ptr.get_output_shape(),
            .unsqueeze => |ptr| return ptr.get_output_shape(),
            else => {
                std.debug.print("\n\nERROR: get_output_shape() is not available!! \n\n", .{});
                return error.OpNotAvailable;
            },
        }
    }

    pub fn get_output_tensors(self: Op_union) ![]*TensorZant {
        return switch (self) {
            .add => |ptr| try ptr.get_output_tensors(),
            .averagePool => |ptr| try ptr.get_output_tensors(),
            .batchNormalization => |ptr| try ptr.get_output_tensors(),
            .cast => |ptr| try ptr.get_output_tensors(),
            .ceil => |ptr| try ptr.get_output_tensors(),
            .clip => |ptr| try ptr.get_output_tensors(),
            .concat => |ptr| try ptr.get_output_tensors(),
            .constant => |ptr| try ptr.get_output_tensors(),
            .conv => |ptr| try ptr.get_output_tensors(),
            .convClip => |ptr| try ptr.get_output_tensors(),
            .convInteger => |ptr| try ptr.get_output_tensors(),

            .dequantizeLinear => |ptr| return ptr.get_output_tensors(),
            .div => |ptr| try ptr.get_output_tensors(),
            .dynamicQuantizeLinear => |ptr| try ptr.get_output_tensors(),
            .elu => |ptr| try ptr.get_output_tensors(),
            .exp => |ptr| try ptr.get_output_tensors(),
            .flatten => |ptr| try ptr.get_output_tensors(),
            .floor => |ptr| try ptr.get_output_tensors(),
            .gather => |ptr| try ptr.get_output_tensors(),
            .gatherND => |ptr| try ptr.get_output_tensors(),
            .gemm => |ptr| try ptr.get_output_tensors(),
            .gelu => |ptr| try ptr.get_output_tensors(),
            .globalAveragePool => |ptr| try ptr.get_output_tensors(),
            .identity => |ptr| try ptr.get_output_tensors(),
            .leakyRelu => |ptr| try ptr.get_output_tensors(),
            .matMul => |ptr| try ptr.get_output_tensors(),
            .maxPool => |ptr| try ptr.get_output_tensors(),
            .min => |ptr| try ptr.get_output_tensors(),
            .mul => |ptr| try ptr.get_output_tensors(),
            .neg => |ptr| try ptr.get_output_tensors(),
            .nonMaxSuppression => |ptr| try ptr.get_output_tensors(),
            .oneHot => |ptr| try ptr.get_output_tensors(),
            .pad => |ptr| try ptr.get_output_tensors(),
            .qgemm => |ptr| try ptr.get_output_tensors(),
            .qlinearadd => |ptr| try ptr.get_output_tensors(),
            .qlinearaveragepool => |ptr| try ptr.get_output_tensors(),
            .qlinearconcat => |ptr| try ptr.get_output_tensors(),
            .qlinearconv => |ptr| try ptr.get_output_tensors(),
            .qlinearglobalaveragepool => |ptr| try ptr.get_output_tensors(),
            .qlinearmatmul => |ptr| try ptr.get_output_tensors(),
            .qlinearmul => |ptr| try ptr.get_output_tensors(),
            .qlinearsoftmax => |ptr| try ptr.get_output_tensors(),
            .quantizeLinear => |ptr| try ptr.get_output_tensors(),
            .reduceMean => |ptr| try ptr.get_output_tensors(),
            .relu => |ptr| try ptr.get_output_tensors(),
            .reshape => |ptr| try ptr.get_output_tensors(),
            .resize => |ptr| try ptr.get_output_tensors(),
            .shape => |ptr| try ptr.get_output_tensors(),
            .sigmoid => |ptr| try ptr.get_output_tensors(),
            .slice => |ptr| try ptr.get_output_tensors(),
            .softmax => |ptr| try ptr.get_output_tensors(),
            .split => |ptr| try ptr.get_output_tensors(),
            .squeeze => |ptr| try ptr.get_output_tensors(),
            .sqrt => |ptr| try ptr.get_output_tensors(),
            .sub => |ptr| try ptr.get_output_tensors(),
            .topK => |ptr| try ptr.get_output_tensors(),
            .tanh => |ptr| try ptr.get_output_tensors(),
            .transpose => |ptr| try ptr.get_output_tensors(),
            .unsqueeze => |ptr| try ptr.get_output_tensors(),
            .useless => |ptr| try ptr.get_output_tensors(),
        };
    }

    pub fn get_input_tensors(self: Op_union) ![]*TensorZant {
        return switch (self) {
            .add => |ptr| try ptr.get_input_tensors(),
            .averagePool => |ptr| try ptr.get_input_tensors(),
            .batchNormalization => |ptr| try ptr.get_input_tensors(),
            .cast => |ptr| try ptr.get_input_tensors(),
            .ceil => |ptr| try ptr.get_input_tensors(),
            .clip => |ptr| try ptr.get_input_tensors(),
            .concat => |ptr| try ptr.get_input_tensors(),
            .constant => |ptr| try ptr.get_input_tensors(),
            .conv => |ptr| try ptr.get_input_tensors(),
            .convClip => |ptr| try ptr.get_input_tensors(),
            .convInteger => |ptr| try ptr.get_input_tensors(),

            .dequantizeLinear => |ptr| return ptr.get_input_tensors(),
            .div => |ptr| try ptr.get_input_tensors(),
            .dynamicQuantizeLinear => |ptr| try ptr.get_input_tensors(),
            .elu => |ptr| try ptr.get_input_tensors(),
            .exp => |ptr| try ptr.get_input_tensors(),
            .flatten => |ptr| try ptr.get_input_tensors(),
            .floor => |ptr| try ptr.get_input_tensors(),
            .gather => |ptr| try ptr.get_input_tensors(),
            .gatherND => |ptr| try ptr.get_input_tensors(),
            .gemm => |ptr| try ptr.get_input_tensors(),
            .gelu => |ptr| try ptr.get_input_tensors(),
            .globalAveragePool => |ptr| try ptr.get_input_tensors(),
            .identity => |ptr| try ptr.get_input_tensors(),
            .leakyRelu => |ptr| try ptr.get_input_tensors(),
            .matMul => |ptr| try ptr.get_input_tensors(),
            .maxPool => |ptr| try ptr.get_input_tensors(),
            .min => |ptr| try ptr.get_input_tensors(),
            .mul => |ptr| try ptr.get_input_tensors(),
            .neg => |ptr| try ptr.get_input_tensors(),
            .nonMaxSuppression => |ptr| try ptr.get_input_tensors(),
            .oneHot => |ptr| try ptr.get_input_tensors(),
            .pad => |ptr| try ptr.get_input_tensors(),
            .qgemm => |ptr| try ptr.get_input_tensors(),
            .qlinearadd => |ptr| try ptr.get_input_tensors(),
            .qlinearaveragepool => |ptr| try ptr.get_input_tensors(),
            .qlinearconcat => |ptr| try ptr.get_input_tensors(),
            .qlinearconv => |ptr| try ptr.get_input_tensors(),
            .qlinearglobalaveragepool => |ptr| try ptr.get_input_tensors(),
            .qlinearmatmul => |ptr| try ptr.get_input_tensors(),
            .qlinearmul => |ptr| try ptr.get_input_tensors(),
            .qlinearsoftmax => |ptr| try ptr.get_input_tensors(),
            .quantizeLinear => |ptr| try ptr.get_input_tensors(),
            .reduceMean => |ptr| try ptr.get_input_tensors(),
            .relu => |ptr| try ptr.get_input_tensors(),
            .reshape => |ptr| try ptr.get_input_tensors(),
            .resize => |ptr| try ptr.get_input_tensors(),
            .shape => |ptr| try ptr.get_input_tensors(),
            .sigmoid => |ptr| try ptr.get_input_tensors(),
            .slice => |ptr| try ptr.get_input_tensors(),
            .softmax => |ptr| try ptr.get_input_tensors(),
            .split => |ptr| try ptr.get_input_tensors(),
            .squeeze => |ptr| try ptr.get_input_tensors(),
            .sqrt => |ptr| try ptr.get_input_tensors(),
            .sub => |ptr| try ptr.get_input_tensors(),
            .topK => |ptr| try ptr.get_input_tensors(),
            .tanh => |ptr| try ptr.get_input_tensors(),
            .transpose => |ptr| try ptr.get_input_tensors(),
            .unsqueeze => |ptr| try ptr.get_input_tensors(),
            .useless => |ptr| try ptr.get_input_tensors(),
        };
    }

    pub fn get_memory_footprint(self: Op_union) !usize {
        const input_tensors = try self.get_input_tensors();
        const output_tensors = try self.get_output_tensors();

        var node_mem: usize = 0;
        for (input_tensors) |t| {
            node_mem += t.getSize() * @sizeOf(t.ty);
        }

        for (output_tensors) |t| {
            node_mem += t.getSize() * @sizeOf(t.ty);
        }

        return node_mem;
    }

    pub fn write_op(self: Op_union, writer: std.fs.File.Writer) !void {
        switch (self) {
            .add => |ptr| try ptr.write_op(writer),
            .averagePool => |ptr| try ptr.write_op(writer),
            .batchNormalization => |ptr| try ptr.write_op(writer),
            .cast => |ptr| try ptr.write_op(writer),
            .ceil => |ptr| try ptr.write_op(writer),
            .clip => |ptr| try ptr.write_op(writer),
            .concat => |ptr| try ptr.write_op(writer),
            .constant => |ptr| try ptr.write_op(writer),
            .conv => |ptr| try ptr.write_op(writer),
            .convClip => |ptr| try ptr.write_op(writer),
            .convInteger => |ptr| try ptr.write_op(writer),

            .dequantizeLinear => |ptr| try ptr.write_op(writer),
            .div => |ptr| try ptr.write_op(writer),
            .dynamicQuantizeLinear => |ptr| try ptr.write_op(writer),
            .elu => |ptr| try ptr.write_op(writer),
            .exp => |ptr| try ptr.write_op(writer),
            .flatten => |ptr| try ptr.write_op(writer),
            .floor => |ptr| try ptr.write_op(writer),
            .gather => |ptr| try ptr.write_op(writer),
            .gatherND => |ptr| try ptr.write_op(writer),
            .gemm => |ptr| try ptr.write_op(writer),
            .gelu => |ptr| try ptr.write_op(writer),
            .globalAveragePool => |ptr| try ptr.write_op(writer),
            .identity => |ptr| try ptr.write_op(writer),
            .leakyRelu => |ptr| try ptr.write_op(writer),
            .matMul => |ptr| try ptr.write_op(writer),
            .maxPool => |ptr| try ptr.write_op(writer),
            .min => |ptr| try ptr.write_op(writer),
            .mul => |ptr| try ptr.write_op(writer),
            .neg => |ptr| try ptr.write_op(writer),
            .nonMaxSuppression => |ptr| try ptr.write_op(writer),
            .oneHot => |ptr| try ptr.write_op(writer),
            .pad => |ptr| try ptr.write_op(writer),
            .qgemm => |ptr| try ptr.write_op(writer),
            .qlinearadd => |ptr| try ptr.write_op(writer),
            .qlinearaveragepool => |ptr| try ptr.write_op(writer),
            .qlinearconcat => |ptr| try ptr.write_op(writer),
            .qlinearconv => |ptr| try ptr.write_op(writer),
            .qlinearglobalaveragepool => |ptr| try ptr.write_op(writer),
            .qlinearmatmul => |ptr| try ptr.write_op(writer),
            .qlinearmul => |ptr| try ptr.write_op(writer),
            .qlinearsoftmax => |ptr| try ptr.write_op(writer),
            .quantizeLinear => |ptr| try ptr.write_op(writer),
            .reduceMean => |ptr| try ptr.write_op(writer),
            .relu => |ptr| try ptr.write_op(writer),
            .reshape => |ptr| try ptr.write_op(writer),
            .resize => |ptr| try ptr.write_op(writer),
            .shape => |ptr| try ptr.write_op(writer),
            .sigmoid => |ptr| try ptr.write_op(writer),
            .slice => |ptr| try ptr.write_op(writer),
            .softmax => |ptr| try ptr.write_op(writer),
            .split => |ptr| try ptr.write_op(writer), //not working! error: .FAULT => unreachable,
            .squeeze => |ptr| try ptr.write_op(writer),
            .sqrt => |ptr| try ptr.write_op(writer),
            .sub => |ptr| try ptr.write_op(writer),
            .topK => |ptr| try ptr.write_op(writer),
            .tanh => |ptr| try ptr.write_op(writer),
            .transpose => |ptr| try ptr.write_op(writer),
            .unsqueeze => |ptr| try ptr.write_op(writer),
            .useless => |ptr| try ptr.write_op(writer),
        }
    }

    pub fn print(self: Op_union) !void {
        switch (self) {
            .add => |ptr| ptr.print(),
            .averagePool => |ptr| ptr.print(),
            .batchNormalization => |ptr| ptr.print(),
            .cast => |ptr| ptr.print(),
            .ceil => |ptr| ptr.print(),
            .clip => |ptr| ptr.print(),
            .concat => |ptr| ptr.print(),
            .constant => |ptr| ptr.print(),
            .conv => |ptr| ptr.print(),
            .convClip => |ptr| ptr.print(),
            .convInteger => |ptr| ptr.print(),
            .dequantizeLinear => |ptr| try ptr.print(),
            .div => |ptr| ptr.print(),
            .dynamicQuantizeLinear => |ptr| ptr.print(),
            .elu => |ptr| ptr.print(),
            .exp => |ptr| ptr.print(),
            .flatten => |ptr| ptr.print(),
            .floor => |ptr| ptr.print(),
            .gather => |ptr| ptr.print(),
            .gatherND => |ptr| ptr.print(),
            .gemm => |ptr| ptr.print(),
            .gelu => |ptr| ptr.print(),
            .globalAveragePool => |ptr| ptr.print(),
            .identity => |ptr| ptr.print(),
            .leakyRelu => |ptr| ptr.print(),
            .matMul => |ptr| ptr.print(),
            .maxPool => |ptr| ptr.print(),
            .min => |ptr| ptr.print(),
            .mul => |ptr| ptr.print(),
            .neg => |ptr| ptr.print(),
            .nonMaxSuppression => |ptr| ptr.print(),
            .oneHot => |ptr| ptr.print(),
            .pad => |ptr| ptr.print(),
            .qgemm => |ptr| ptr.print(),
            .qlinearadd => |ptr| ptr.print(),
            .qlinearaveragepool => |ptr| ptr.print(),
            .qlinearconcat => |ptr| ptr.print(),
            .qlinearconv => |ptr| try ptr.print(),
            .qlinearglobalaveragepool => |ptr| ptr.print(),
            .qlinearmatmul => |ptr| ptr.print(),
            .qlinearmul => |ptr| ptr.print(),
            .qlinearsoftmax => |ptr| ptr.print(),
            .quantizeLinear => |ptr| try ptr.print(),
            .reduceMean => |ptr| ptr.print(),
            .relu => |ptr| ptr.print(),
            .reshape => |ptr| ptr.print(),
            .resize => |ptr| ptr.print(),
            .shape => |ptr| ptr.print(),
            .sigmoid => |ptr| ptr.print(),
            .slice => |ptr| ptr.print(),
            .softmax => |ptr| ptr.print(),
            .split => |ptr| ptr.print(),
            .squeeze => |ptr| ptr.print(),
            .sqrt => |ptr| ptr.print(),
            .sub => |ptr| ptr.print(),
            .topK => |ptr| ptr.print(),
            .tanh => |ptr| ptr.print(),
            .transpose => |ptr| ptr.print(),
            .unsqueeze => |ptr| ptr.print(),
            else => {
                std.debug.print("\n\nERROR: print() is not available!! \n\n", .{});
                return error.print_notAvailable;
            },
        }
    }

    /// Render the lower-level math operation for the operator.
    /// DEPRECATED, TO BE REMOVED IN FUTURE VERSIONS
    pub fn render_lower_math_op(self: Op_union, builder: *UOpBuilder) !void {
        switch (self) {
            .add => |ptr| ptr.render_lower(builder),
            .ceil => |ptr| ptr.render_lower(builder),
            .conv => |ptr| ptr.render_lower(builder),
            .identity => |ptr| ptr.render_lower(builder),
            .div => |ptr| ptr.render_lower(builder),
            .identity => |ptr| ptr.render_lower(builder),
            .matMul => |ptr| ptr.render_lower(builder),
            .maxPool => |ptr| ptr.render_lower(builder),
            .mul => |ptr| ptr.render_lower(builder),
            .neg => |ptr| ptr.render_lower(builder),
            .relu => |ptr| ptr.render_lower(builder),
            .reshape => |ptr| ptr.render_lower(builder),
            .tanh => |ptr| ptr.render_lower(builder),
            else => {
                std.debug.print("\n\nERROR: render_lower() is not available!! \n\n", .{});
                return error.print_notAvailable;
            },
        }
    }
};
