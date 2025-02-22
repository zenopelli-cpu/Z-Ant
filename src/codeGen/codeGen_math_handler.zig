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
const codegen_options = @import("codegen_options");

// ----------------------------------- MATH -----------------------------------

/// This method map and write the ONNX operations with the Zant LeanTensorMath mathods
/// Follow the link for details: https://onnx.ai/onnx/operators/?utm_source=chatgpt.com
pub fn write_math_op(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    if (!codegen_options.noComm) {
        try write_op_info(writer, node);
    }
    if (codegen_options.log) {
        try writer.print(
            \\ 
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Running {s} operation...\n")));
            \\    }}
        , .{node.*.nodeProto.*.op_type});
    }

    if (std.mem.eql(u8, node.nodeProto.op_type, "Add")) {
        try write_add(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "AveragePool")) {
        try writer.writeAll("// Handle AveragePool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "BatchNormalization")) {
        try writer.writeAll("// Handle BatchNormalization\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Concat")) {
        try writer.writeAll("// Handle Concat\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Constant")) {
        try writer.writeAll("// Handle Constant\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Conv")) {
        try write_conv(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Div")) {
        try write_div(writer, node);
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
        try write_matmul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MaxPool")) {
        try writer.writeAll("// Handle MaxPool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mul")) {
        try write_mul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try writer.writeAll("// Handle OneHot\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try write_ReLU(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try write_reshape(writer, node);
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
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sum")) {
        try write_sum(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Transpose")) {
        try writer.writeAll("// Handle Transpose\n");
    } else {
        return error.OperationNotSupported;
    }

    try writer.writeAll(" catch return;");
}

fn write_op_info(writer: std.fs.File.Writer, node: *ReadyNode) !void {
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
}

inline fn write_add(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Add.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensors_lean(T, &tensor_{s}, @constCast(&tensor_{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_conv(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Conv.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    //      - W (heterogeneous) - T: The weight tensor
    //      - B (optional, heterogeneous) - T: Optional 1D bias to be added to the convolution, has size of M.
    // OUTPUT:
    //      - Y (heterogeneous) - T: Output data tensor that contains the result of the convolution
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET
    //      - dilations - INTS : dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
    //      - group - INT (default is '1'): number of groups input channels and output channels are divided into
    //      - kernel_shape - INTS : The shape of the convolution kernel. If not present, should be inferred from input W
    //      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
    //      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

    var auto_pad: []const u8 = "NOTSET";
    var dilations: ?[]i64 = undefined;
    var group: i64 = 1;
    var kernel_shape: []i64 = undefined;
    var pads: ?[]i64 = null;
    var strides: ?[]i64 = undefined; //mandatory

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
            if (attr.type == AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
        }
    }
    //----create ?bias string
    var bias_string: []u8 = undefined;
    // Bias Tensor B is optional! verify the presence
    if (node.inputs.items.len == 3) {
        const B_name = try utils.getSanitizedName(node.inputs.items[2].name);
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", B_name, ")" });
    } else {
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
    }

    //----create stride string
    if (strides == null) return error.StrideNotFound;
    const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(strides.?);

    //----create ?pads string
    var pads_string: []const u8 = undefined;
    if (pads != null) {
        pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
    } else {
        pads_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    //----create ?dilatations string
    var dilat_string: []const u8 = undefined;
    if (dilations != null) {
        dilat_string = try utils.i64SliceToUsizeArrayString(dilations.?);
    } else {
        dilat_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    // pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void
    _ = try writer.print(
        \\    
        \\    tensMath.conv_lean(
        \\        T, //type
        \\        &tensor_{s}, //input
        \\        @constCast(&tensor_{s}), //kernel
        \\        &tensor_{s}, //output
        \\        {s}, //bias
        \\        {s}, //stride
        \\        {s}, //pads
        \\        {s}, //dilatations
        \\        {}, //group
        \\        "{s}", //auto_pad
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), //Input
        try utils.getSanitizedName(node.inputs.items[1].name), //Kernel
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        bias_string, //Bias
        stride_string, //Strides
        pads_string, //Pads
        dilat_string, //Dilatations
        group, //Group
        auto_pad,
    });
}

inline fn write_div(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Div.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.div_lean(T, &tensor_{s}, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
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
        c_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", C_name, ")" });
    } else {
        c_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    _ = try writer.print(
        \\
        \\    tensMath.gemm_lean(T, &tensor_{s}, @constCast(&tensor_{s}), {s}, {}, {}, {}, {}, &tensor_{s} )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        c_tensor_string,
        alpha,
        beta,
        transA,
        transB,
        try utils.getSanitizedName(node.outputs.items[0].name), // Output
    });
}

inline fn write_matmul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__MatMul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.matmul_lean(T, &tensor_{s}, @constCast(&tensor_{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_mul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Mul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    _ = try writer.print(
        \\
        \\    tensMath.mul_lean(T, &tensor_{s}, @constCast(&tensor_{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_ReLU(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    tensMath.ReLU_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_reshape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Reshape.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    //      - shape (heterogeneous) - tensor(int64): Specified shape for output.
    // OUTPUTS:
    //      - reshaped (heterogeneous) - T: Reshaped data.
    // ATTRIBUTES:
    //      - allowzero - INT (default is '0'): Whether to allow zeros in shape tensor

    var allowzer: bool = false;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "allowzero")) |_| {
            if (attr.type == AttributeType.INT) allowzer = attr.i != 0;
        }
    }

    const shape_name = try utils.getSanitizedName(node.inputs.items[1].name);
    _ = try writer.print(
        \\
        \\    var shape_usize_{s} = [_]usize{{ for (tensor_{s}.data) |v| @as(usize, @intCast(v)) }};
        \\    tensMath.reshape_lean(T, @constCast(&tensor_{s}), &shape_usize, {}, &tensor_{s})
    , .{
        shape_name,
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor
        allowzer,
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_sigmoid(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\    tensMath.sigmoid_lean(T, &tensor_{s}, &tensor_{s})
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
        \\    tensMath.softmax_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_sum(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Sum.html
    // INPUTS:
    //      - list of tensors
    // OUTPUTS:
    //      - sum (heterogeneous) - T: Output tensor.

    //Writing the tensor list with all the inputs
    _ = try writer.print(
        \\
        \\    const my_tensor_list = [_]*Tensor(T){{
    , .{});

    for (node.inputs.items, 0..) |tens, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }
        _ = try writer.print(
            \\tensor_{s}
        , .{try utils.getSanitizedName(tens.name)});
    }

    _ = try writer.print("}}", .{});

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensor_list_lean(T, T, &my_tensor_list, &tensor_{s})
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});
}

// ----------------------------------- SHAPE inference -----------------------------------

pub fn compute_output_shape(readyNode: *ReadyNode) !void {
    if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Add")) {
        //https://onnx.ai/onnx/operators/onnx__Add.html
        readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Concat")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Constant")) {
        //https://onnx.ai/onnx/operators/onnx__Constant.html
        try compute_constant_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Conv")) {
        //https://onnx.ai/onnx/operators/onnx__Conv.html
        try compute_conv_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Div")) {
        //https://onnx.ai/onnx/operators/onnx__Div.html
        readyNode.outputs.items[0].shape = readyNode.inputs.items[1].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Flatten")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gather")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gemm")) {
        //https://onnx.ai/onnx/operators/onnx__Gemm.html
        try compute_gemm_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LeakyRelu")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LogSoftmax")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MatMul")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MaxPool")) {
        try compute_maxPool_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mul")) {
        //https://onnx.ai/onnx/operators/onnx__Mul.html
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "OneHot")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Relu")) {
        //https://onnx.ai/onnx/operators/onnx__Relu.html
        try compute_ReLU_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Reshape")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Resize")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Shape")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sigmoid")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Softmax")) {
        //https://onnx.ai/onnx/operators/onnx__Softmax.html
        try compute_softmax_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Slice")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Split")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sub")) {
        // TODO
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Transpose")) {
        // TODO
    } else {
        std.debug.print("\n\n ERROR! output shape computation for {s} is not available in codeGen_math_handler.compute_output_shape() \n\n", .{readyNode.nodeProto.op_type});
        return error.OperationNotSupported;
    }
}

// ---------------- SHAPE COMPUTATION METHODS ----------------

inline fn compute_constant_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_constant_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const shape = try utils.getConstantTensorDims(readyNode.nodeProto);
    std.debug.print("\n output_shape: []i64 = {any}", .{shape});
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_ReLU_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_ReLU_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_softmax_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_softmax_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_gemm_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_gemm_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    std.debug.print("\n weight_shape: []i64 = {any}", .{readyNode.inputs.items[1].shape});
    std.debug.print("\n bias_shape: []i64 = {any}", .{readyNode.inputs.items[2].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[2].shape;
}

inline fn compute_mul_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_mul_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_a_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    std.debug.print("\n input_b_shape: []i64 = {any}", .{readyNode.inputs.items[1].shape});

    const shape_len = readyNode.outputs.items[0].shape.len;
    var newShape = try allocator.alloc(i64, shape_len);
    @memcpy(newShape, readyNode.inputs.items[0].shape);
    newShape[shape_len - 1] = readyNode.inputs.items[1].shape[shape_len - 1];
    readyNode.outputs.items[0].shape = newShape;
}

inline fn compute_conv_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_conv_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;
    const kernel_shape: []const i64 = readyNode.inputs.items[1].shape;
    const stride = readyNode.nodeProto.attribute[1].ints;

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape});
    std.debug.print("\n stride: []i64 = {any}", .{stride});

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_convolution_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                try utils.i64SliceToUsizeSlice(kernel_shape),
                try utils.i64SliceToUsizeSlice(stride),
            ),
        ),
    );
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_maxPool_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_maxPool_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;
    const kernel_shape = readyNode.nodeProto.attribute[0].ints;
    const stride = readyNode.nodeProto.attribute[1].ints;

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape});
    std.debug.print("\n stride: []i64 = {any}", .{stride});

    const kernel_2d = [2]usize{ @intCast(kernel_shape[0]), @intCast(kernel_shape[1]) };
    const stride_2d = [2]usize{ @intCast(stride[0]), @intCast(stride[1]) };

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_pooling_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                kernel_2d,
                stride_2d,
            ),
        ),
    );
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}
