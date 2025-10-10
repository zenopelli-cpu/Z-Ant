//! Stop whatever you are doing and read this before proceding!
//! https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
//!
//!
//!
//!  At the moment Zant supports IR_VERSION_2024_3_25 !!!  date: 31/08/2025
const std = @import("std");
const protobuf = @import("protobuf.zig");

pub const ValueInfoProto = @import("valueInfoProto.zig").ValueInfoProto;
pub const AttributeProto = @import("attributeProto.zig").AttributeProto;
pub const TensorShapeProto = @import("tensorShapeProto.zig").TensorShapeProto;
pub const TypeProto = @import("typeProto.zig").TypeProto;
pub const TensorProto = @import("tensorProto.zig").TensorProto;
pub const NodeProto = @import("nodeProto.zig").NodeProto;
pub const GraphProto = @import("graphProto.zig").GraphProto;
pub const ModelProto = @import("modelProto.zig").ModelProto;
pub const StringStringEntryProto = @import("stringStringEntryProto.zig").StringStringEntryProto;
pub const OperatorSetIdProto = @import("operatorSetIdProto.zig").OperatorSetIdProto;
pub const FunctionProto = @import("functionProto.zig").FunctionProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

const onnx_log = std.log.scoped(.onnx);

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
    IR_VERSION_2025_05_12 = 0x000000000000000B,
    IR_VERSION = 0x000000000000000C,
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

pub const DataLocation = enum(i32) {
    DEFAULT = 0,
    EXTERNAL = 1,
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

pub const OnnxOperator = enum {
    ADD,
    AVERAGEPOOL,
    BATCHNORMALIZATION,
    CAST,
    CEIL,
    CLIP,
    CONCAT,
    CONSTANT,
    CONV,
    CONVINTEGER,
    DEQUANTIZE,
    DEQUANTIZELINEAR,
    DIV,
    DYNAMICQUANTIZELINEAR,
    ELU,
    EXP,
    FLATTEN,
    FLOOR,
    GATHER,
    GATHERND,
    GELU,
    GEMM,
    GLOBALAVERAGEPOOL,
    IDENTITY,
    LEAKYRELU,
    MATMUL,
    MAXPOOL,
    MIN,
    MUL,
    NEG,
    NONMAXSUPPRESSION,
    ONEHOT,
    PAD,
    QGEMM,
    QLINEARADD,
    QLINEARAVERAGEPOOL,
    QLINEARCONCAT,
    QLINEARCONV,
    QLINEARGLOBALAVERAGEPOOL,
    QLINEARMATMUL,
    QLINEARMUL,
    QLINEARSOFTMAX,
    QUANTIZE,
    QUANTIZELINEAR,
    REDUCEMEAN,
    RELU,
    RESHAPE,
    RESIZE,
    SHAPE,
    SIGMOID,
    SLICE,
    SOFTMAX,
    SPLIT,
    SQRT,
    SQUEEZE,
    SUB,
    TANH,
    TOPK,
    TRANSPOSE,
    UNSQUEEZE,
    USELESS,
};

pub fn isQlinear(op_type: OnnxOperator) bool {
    return op_type == .QLINEARADD or op_type == .QLINEARAVERAGEPOOL or op_type == .QLINEARCONCAT or op_type == .QLINEARCONV or op_type == .QLINEARGLOBALAVERAGEPOOL or op_type == .QLINEARMATMUL or op_type == .QLINEARMUL or op_type == .QLINEARSOFTMAX;
}

pub fn fromString(op_type: []const u8) !OnnxOperator {
    if (std.mem.eql(u8, op_type, "Add")) return .ADD;
    if (std.mem.eql(u8, op_type, "AveragePool")) return .AVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "BatchNormalization")) return .BATCHNORMALIZATION;
    if (std.mem.eql(u8, op_type, "Cast")) return .CAST;
    if (std.mem.eql(u8, op_type, "Ceil")) return .CEIL;
    if (std.mem.eql(u8, op_type, "Clip")) return .CLIP;
    if (std.mem.eql(u8, op_type, "Concat")) return .CONCAT;
    if (std.mem.eql(u8, op_type, "Constant")) return .CONSTANT;
    if (std.mem.eql(u8, op_type, "Conv")) return .CONV;
    if (std.mem.eql(u8, op_type, "ConvInteger")) return .CONVINTEGER;
    if (std.mem.eql(u8, op_type, "Dequantize")) return .DEQUANTIZE;
    if (std.mem.eql(u8, op_type, "DequantizeLinear")) return .DEQUANTIZELINEAR;
    if (std.mem.eql(u8, op_type, "Div")) return .DIV;
    if (std.mem.eql(u8, op_type, "DynamicQuantizeLinear")) return .DYNAMICQUANTIZELINEAR;
    if (std.mem.eql(u8, op_type, "Elu")) return .ELU;
    if (std.mem.eql(u8, op_type, "Exp")) return .EXP;
    if (std.mem.eql(u8, op_type, "Flatten")) return .FLATTEN;
    if (std.mem.eql(u8, op_type, "Floor")) return .FLOOR;
    if (std.mem.eql(u8, op_type, "Gather")) return .GATHER;
    if (std.mem.eql(u8, op_type, "GatherND")) return .GATHERND;
    if (std.mem.eql(u8, op_type, "Gelu")) return .GELU;
    if (std.mem.eql(u8, op_type, "Gemm")) return .GEMM;
    if (std.mem.eql(u8, op_type, "GlobalAveragePool")) return .GLOBALAVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "Identity")) return .IDENTITY;
    if (std.mem.eql(u8, op_type, "LeakyRelu")) return .LEAKYRELU;
    if (std.mem.eql(u8, op_type, "MatMul")) return .MATMUL;
    if (std.mem.eql(u8, op_type, "MaxPool")) return .MAXPOOL;
    if (std.mem.eql(u8, op_type, "Min")) return .MIN;
    if (std.mem.eql(u8, op_type, "Mul")) return .MUL;
    if (std.mem.eql(u8, op_type, "Neg")) return .NEG;
    if (std.mem.eql(u8, op_type, "NonMaxSuppression")) return .NONMAXSUPPRESSION;
    if (std.mem.eql(u8, op_type, "OneHot")) return .ONEHOT;
    if (std.mem.eql(u8, op_type, "Pad")) return .PAD;
    if (std.mem.eql(u8, op_type, "QGemm")) return .QGEMM;
    if (std.mem.eql(u8, op_type, "QLinearAdd")) return .QLINEARADD;
    if (std.mem.eql(u8, op_type, "QLinearAveragePool")) return .QLINEARAVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "QLinearConcat")) return .QLINEARCONCAT;
    if (std.mem.eql(u8, op_type, "QLinearConv")) return .QLINEARCONV;
    if (std.mem.eql(u8, op_type, "QLinearGlobalAveragePool")) return .QLINEARGLOBALAVERAGEPOOL;
    if (std.mem.eql(u8, op_type, "QLinearMatMul")) return .QLINEARMATMUL;
    if (std.mem.eql(u8, op_type, "QLinearMul")) return .QLINEARMUL;
    if (std.mem.eql(u8, op_type, "QLinearSoftmax")) return .QLINEARSOFTMAX;
    if (std.mem.eql(u8, op_type, "Quantize")) return .QUANTIZE;
    if (std.mem.eql(u8, op_type, "QuantizeLinear")) return .QUANTIZELINEAR;
    if (std.mem.eql(u8, op_type, "ReduceMean")) return .REDUCEMEAN;
    if (std.mem.eql(u8, op_type, "Relu")) return .RELU;
    if (std.mem.eql(u8, op_type, "Reshape")) return .RESHAPE;
    if (std.mem.eql(u8, op_type, "Resize")) return .RESIZE;
    if (std.mem.eql(u8, op_type, "Shape")) return .SHAPE;
    if (std.mem.eql(u8, op_type, "Sigmoid")) return .SIGMOID;
    if (std.mem.eql(u8, op_type, "Slice")) return .SLICE;
    if (std.mem.eql(u8, op_type, "Softmax")) return .SOFTMAX;
    if (std.mem.eql(u8, op_type, "Split")) return .SPLIT;
    if (std.mem.eql(u8, op_type, "Sqrt")) return .SQRT;
    if (std.mem.eql(u8, op_type, "Squeeze")) return .SQUEEZE;
    if (std.mem.eql(u8, op_type, "Sub")) return .SUB;
    if (std.mem.eql(u8, op_type, "Tanh")) return .TANH;
    if (std.mem.eql(u8, op_type, "TopK")) return .TOPK;
    if (std.mem.eql(u8, op_type, "Transpose")) return .TRANSPOSE;
    if (std.mem.eql(u8, op_type, "Unsqueeze")) return .UNSQUEEZE;
    if (std.mem.eql(u8, op_type, "Useless")) return .USELESS;

    return error.UnknownOperator;
}

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

    if (model.graph.?.value_info.len == 0 and model.graph.?.nodes.len > 1) {
        std.debug.print("\n\n\n+-------------------------------------------+ ", .{});
        std.debug.print("\n   Your model do not contains intermediate tensor shapes,\n   run ' python3 src/onnx/shape_thief.py --model modelName '", .{});
        std.debug.print("\n+-------------------------------------------+ \n\n", .{});

        std.debug.print("\n\n+-------------------------------------------+ ", .{});
        std.debug.print("\n   Also ensure that the input shape is well known, otherwise: \n   run ' python3 src/onnx/input_setter.py --model modelName --shape B,C,H,W (eg., \"1,3,10,10\")'", .{});
        std.debug.print("\n+-------------------------------------------+ \n\n\n", .{});

        unreachable;
    }

    return model;
}

// calculates the size in bytes of a tensor based on its data type and dimensions
pub fn tensorSizeInBytes(tensor: *const TensorProto) usize {
    var num_elements: usize = 1;
    for (tensor.dims) |dim| {
        num_elements *= @intCast(dim);
    }

    const type_size: usize = switch (tensor.data_type) {
        .UINT8, .INT8, .BOOL, .FLOAT8E4M3FN, .FLOAT8E4M3FNUZ, .FLOAT8E5M2, .FLOAT8E5M2FNUZ => 1, // 1-byte types
        .UINT16, .INT16, .FLOAT16, .BFLOAT16 => 2, // 2-byte types
        .FLOAT, .INT32, .UINT32 => 4, // 4-byte types
        .DOUBLE, .INT64, .UINT64, .COMPLEX64 => 8, // 8-byte types
        .COMPLEX128 => 16, // 16-byte types
        else => {
            onnx_log.warn("Warning: Unknown data type {} in tensor, assuming 4 bytes per element\n", .{tensor.data_type});
            return 4 * num_elements; // Default to 4 bytes for unknown types
        },
    };

    return num_elements * type_size;
}

pub fn printModelDetails(model: *const ModelProto) !void {
    const stdout = std.debug;

    // basic model's informations
    stdout.print("\n=========== ONNX Model Details ===========\n", .{});
    stdout.print("Model version: {}\n", .{model.ir_version});
    stdout.print("Producer: {s}\n", .{model.producer_name orelse "Unknown"});

    // graph informations
    if (model.graph) |graph| {
        stdout.print("\nGraph Statistics:\n", .{});
        stdout.print("  Number of nodes: {}\n", .{graph.nodes.len});

        // operator count
        var op_counts = std.StringHashMap(usize).init(std.heap.page_allocator);
        defer op_counts.deinit();
        for (graph.nodes) |node| {
            const op_type = node.op_type;
            const count = op_counts.get(op_type) orelse 0;
            try op_counts.put(op_type, count + 1);
        }
        stdout.print("  Operator distribution:\n", .{});
        var op_iter = op_counts.iterator();
        while (op_iter.next()) |entry| {
            stdout.print("    {s}: {}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        // Tensors and weights
        var tensor_count: usize = 0;
        for (graph.initializers) |_| tensor_count += 1;
        for (graph.inputs) |_| tensor_count += 1;
        for (graph.outputs) |_| tensor_count += 1;

        var total_weight_size: usize = 0;
        for (graph.initializers) |tensor| {
            total_weight_size += tensorSizeInBytes(tensor);
        }

        stdout.print("\nMemory Requirements:\n", .{});
        stdout.print("  Total tensors: {}\n", .{tensor_count});
        stdout.print("  Total weight size: {} bytes ({d:.2} MB)\n", .{ total_weight_size, @as(f32, @floatFromInt(total_weight_size)) / (1024.0 * 1024.0) });
    } else {
        stdout.print("\nWARNING: No graph found in the model.\n", .{});
    }

    stdout.print("=========================================\n", .{});
}
