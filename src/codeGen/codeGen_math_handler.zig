const std = @import("std");
const Tensor = @import("tensor").Tensor;
const tensorMath = @import("tensor_math");
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const allocator = @import("pkgAllocator").allocator;

// --- proto libs
const TensorProto = @import("onnx").TensorProto;
const NodeProto = @import("onnx").NodeProto;
const GraphProto = @import("onnx").GraphProto;
const AttributeType = @import("onnx").AttributeType;

// --- codeGen libs
const ReadyNode = @import("codeGen_predict.zig").ReadyNode;
const ReadyTensor = @import("codeGen_predict.zig").ReadyTensor;
const utils = @import("codeGen_utils.zig");

// ----------------------------------- MATH -----------------------------------

/// This method map and write the ONNX operations with the Zant LeanTensorMath mathods
/// Follow the link for details: https://onnx.ai/onnx/operators/?utm_source=chatgpt.com
pub fn write_math_op(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    try writer.print(
        \\
        \\
        \\   //forwarding operation : {s}
        \\   //parameters:
        \\   //   inputs: 
    , .{node.*.nodeProto.*.op_type});

    //write the inputs
    for (node.inputs.items) |input| {
        try writer.print(
            \\
            \\   //      -> {s} 
        , .{input.name});
    }
    try writer.print(
        \\
        \\   //    outputs: 
    , .{});

    //write the outputs
    for (node.outputs.items) |output| {
        try writer.print(
            \\
            \\   //      <- {s} 
        , .{output.name});
    }

    if (std.mem.eql(u8, node.nodeProto.op_type, "Add")) {
        try writer.writeAll("// Handle Add\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "AveragePool")) {
        try writer.writeAll("// Handle AveragePool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "BatchNormalization")) {
        try writer.writeAll("// Handle BatchNormalization\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Concat")) {
        try writer.writeAll("// Handle Concat\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Constant")) {
        try writer.writeAll("// Handle Constant\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Conv")) {
        try writer.writeAll("// Handle Conv\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Div")) {
        try writer.writeAll("// Handle Div\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Flatten")) {
        try writer.writeAll("// Handle Flatten\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gather")) {
        try writer.writeAll("// Handle Gather\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gemm")) {
        try write_gemm(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LeakyRelu")) {
        try writer.writeAll("// Handle LeakyRelu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LogSoftmax")) {
        try writer.writeAll("// Handle LogSoftmax\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MatMul")) {
        try writer.writeAll("// Handle MatMul\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MaxPool")) {
        try writer.writeAll("// Handle MaxPool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mul")) {
        try writer.writeAll("// Handle Mul\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try writer.writeAll("// Handle OneHot\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try write_ReLU(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try writer.writeAll("// Handle Relu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Resize")) {
        try writer.writeAll("// Handle Resize\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sigmoid")) {
        try write_sigmoid(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Softmax")) {
        try write_softmax(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Slice")) {
        try writer.writeAll("// Handle Slice\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Split")) {
        try writer.writeAll("// Handle Split\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sub")) {
        try writer.writeAll("// Handle Sub\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Transpose")) {
        try writer.writeAll("// Handle Transpose\n");
    } else {
        return error.OperationNotSupported;
    }
}

inline fn write_gemm(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Gemm.html
    // INPUTS:
    //      - Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
    //      - Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
    //      - Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
    //OUTPUTS:
    //      - Output tensor of shape (M, N).
    // ATTRIBUTES:
    //      - alpha. FLOAT (default is '1.0'): Scalar multiplier for the product of input tensors A * B.
    //      - beta - FLOAT (default is '1.0'): Scalar multiplier for input tensor C.
    //      - transA - INT (default is '0'): Whether A should be transposed
    //      - transB - INT (default is '0'): Whether B should be transposed

    var alpha: f32 = 1.0;
    var beta: f32 = 1.0;
    var transA: bool = false;
    var transB: bool = false;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "alpha")) |_| {
            if (attr.type == AttributeType.FLOAT) alpha = attr.f else return error.GemmAphaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "beta")) |_| {
            if (attr.type == AttributeType.FLOAT) beta = attr.f else return error.GemmBetaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "transA")) |_| {
            if (attr.type == AttributeType.INT) transA = if (attr.i != 0) false else true else return error.GemmTransANotINT;
        } else if (std.mem.indexOf(u8, attr.name, "transB")) |_| {
            if (attr.type == AttributeType.INT) transB = if (attr.i != 0) false else true else return error.GemmTransBNotINT;
        }
    }

    var c_tensor_string: []u8 = undefined;
    // Input Tensor C is optional! verify the presence
    if (node.inputs.items.len == 3) {
        const C_name = try utils.getSanitizedName(node.inputs.items[2].name);
        c_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ ", &tensor_", C_name });
    } else {
        c_tensor_string = "";
    }

    //gemm_lean(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: ?*const f32, beta: ?*const f32, transA: ?*const bool, transB: ?*const bool, Y: *Tensor(T)
    _ = try writer.print(
        \\
        \\    try tensMath.gemm(T, &tensor_{s}, &tensor_{s} {s}, {}, {}, {}, {} );
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        c_tensor_string,
        alpha,
        beta,
        transA,
        transB,
    });
}

inline fn write_ReLU(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    try tensMath.ReLU(T, &tensor_{s}, &tensor_{s});
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_sigmoid(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    try tensMath.sigmoid(T, &tensor_{s}, &tensor_{s});
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_softmax(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    try tensMath.softmax(T, &tensor_{s}, &tensor_{s});
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

// ----------------------------------- SHAPE inference -----------------------------------

pub fn compute_output_shape(readyNode: *ReadyNode) !void {
    if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Add")) {
        // try writer.writeAll("// Handle Shape Add\n");
        readyNode.outputs.items[0].shape = readyNode.inputs.items[1].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Concat")) {
        // try writer.writeAll("// Handle Shape Concat\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Constant")) {
        // try writer.writeAll("// Handle Shape Constant\n");
        try compute_constant_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Conv")) {
        // try writer.writeAll("// Handle Shape Conv\n");
        try compute_conv_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Div")) {
        // try writer.writeAll("// Handle Shape Div\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Flatten")) {
        // try writer.writeAll("// Handle Shape Flatten\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gather")) {
        // try writer.writeAll("// Handle Shape Gather\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gemm")) {
        // try writer.writeAll("// Handle Shape Gemm\n");
        try compute_gemm_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LeakyRelu")) {
        // try writer.writeAll("// Handle Shape LeakyRelu\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LogSoftmax")) {
        // try writer.writeAll("// Handle Shape LogSoftmax\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MatMul")) {
        // try writer.writeAll("// Handle Shape MatMul\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MaxPool")) {
        // try writer.writeAll("// Handle Shape MaxPool\n");
        //try compute_maxPool_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mul")) {
        // try writer.writeAll("// Handle Shape Mul\n");
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "OneHot")) {
        // try writer.writeAll("// Handle Shape OneHot\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Relu")) {
        // try writer.writeAll("// Handle Shape Relu\n");
        try compute_ReLU_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Reshape")) {
        // try writer.writeAll("// Handle Shape Reshape\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Resize")) {
        // try writer.writeAll("// Handle Shape Resize\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Shape")) {
        // try writer.writeAll("// Handle Shape Shape\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sigmoid")) {
        // try writer.writeAll("// Handle Shape Sigmoid\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Softmax")) {
        // try writer.writeAll("// Handle Shape Softmax\n");
        try compute_softmax_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Slice")) {
        // try writer.writeAll("// Handle Shape Slice\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Split")) {
        // try writer.writeAll("// Handle Shape Split\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sub")) {
        // try writer.writeAll("// Handle Shape Sub\n");
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Transpose")) {
        // try writer.writeAll("// Handle Shape Transpose\n");
    } else {
        std.debug.print("\n\n ERROR! output shape computation for {s} is not available in codeGen_math_handler.compute_output_shape() \n\n", .{readyNode.nodeProto.op_type});
        return error.OperationNotSupported;
    }
}

// ---------------- SHAPE COMPUTATION METHODS ----------------

inline fn compute_constant_output_shape(readyNode: *ReadyNode) !void {
    readyNode.outputs.items[0].shape = try utils.getConstantTensorDims(readyNode.nodeProto);
}

inline fn compute_ReLU_output_shape(readyNode: *ReadyNode) !void {
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_softmax_output_shape(readyNode: *ReadyNode) !void {
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_gemm_output_shape(readyNode: *ReadyNode) !void {
    //inputs.items[0] -> input Tensor
    //inputs.items[1] -> weight Tensor
    //inputs.items[2] -> bias Tensor
    //
    //output shape = bias shape by definition of gemm

    readyNode.outputs.items[0].shape = readyNode.inputs.items[2].shape;
}

inline fn compute_mul_output_shape(readyNode: *ReadyNode) !void {
    //inputs.items[0] ->  Tensor a
    //inputs.items[1] ->  Tensor b
    //
    //output shape =[... , a.rows , b.cols ]

    const shape_len = readyNode.outputs.items[0].shape.len;

    var newShape = try allocator.alloc(i64, shape_len);
    @memcpy(newShape, readyNode.inputs.items[0].shape);
    newShape[shape_len - 1] = readyNode.inputs.items[1].shape[shape_len - 1];

    readyNode.outputs.items[0].shape = newShape;
}

inline fn compute_conv_output_shape(readyNode: *ReadyNode) !void {
    //inputs.items[0] -> input Tensor (X)
    //inputs.items[1] -> kernel Tensor (W)
    //
    //output shape-> input Tensor

    //attributes:
    //nodeProtop.attribute[0] = kernel_shape -> TODO: search it, it is not fixed to index 0
    //nodeProtop.attribute[1] = strides -> TODO: search it, it is not fixed to index 1

    const input_shape: []const i64 = readyNode.inputs.items[0].shape;
    const kernel_shape: []const i64 = readyNode.inputs.items[1].shape;
    const stride = readyNode.nodeProto.attribute[1].ints;

    // DEBUG
    std.debug.print("\n====== compute_conv_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []usize = {any}", .{try utils.i64SliceToUsizeSlice(input_shape)});
    std.debug.print("\n kernel_shape: []usize = {any} ", .{try utils.i64SliceToUsizeSlice(kernel_shape)});

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_convolution_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                try utils.i64SliceToUsizeSlice(kernel_shape),
                try utils.i64SliceToUsizeSlice(stride),
            ),
        ),
    );
}

// inline fn compute_maxPool_output_shape(readyNode: *ReadyNode) !void {
//     readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
//         @constCast(
//             &try tensorMath.get_convolution_output_shape(
//                 try utils.i64SliceToUsizeSlice(input_shape),
//                 try utils.i64SliceToUsizeSlice(kernel_shape),
//                 try utils.i64SliceToUsizeSlice(stride),
//             ),
//         ),
//     );
// }
