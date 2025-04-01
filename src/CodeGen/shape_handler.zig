const std = @import("std");
const os = std.os;

const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const allocator = zant.utils.allocator.allocator;

// --- proto libs
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeType = onnx.AttributeType;

// --- codeGen libs
const ReadyNode = @import("globals.zig").ReadyNode;
const ReadyTensor = @import("globals.zig").ReadyTensor;
const codegen = @import("codegen.zig");
const utils = codegen.utils;
const codegen_options = @import("codegen_options");
const globals = @import("globals.zig");

// ----------------------------------- SHAPE inference -----------------------------------

pub fn compute_output_shape(readyNode: *ReadyNode) !void {
    if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Add")) {
        //https://onnx.ai/onnx/operators/onnx__Add.html
        try compute_Add_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Ceil")) {
        //https://onnx.ai/onnx/operators/onnx__Ceil.html
        try compute_ceil_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Concat")) {
        //https://onnx.ai/onnx/operators/onnx__Concat.html
        try compute_concat_output_shape(readyNode);
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
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gather")) {
        try compute_gather_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gemm")) {
        //https://onnx.ai/onnx/operators/onnx__Gemm.html
        try compute_gemm_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LeakyRelu")) {
        try compute_leaky_relu_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LogSoftmax")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MatMul")) {
        try compute_matmul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MaxPool")) {
        try compute_maxPool_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mul")) {
        //https://onnx.ai/onnx/operators/onnx__Mul.html
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Neg")) {
        //https://onnx.ai/onnx/operators/onnx__Neg.html
        try compute_neg_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "OneHot")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "ReduceMean")) {
        try compute_reducemean_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Relu")) {
        //https://onnx.ai/onnx/operators/onnx__Relu.html
        try compute_ReLU_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Reshape")) {
        // https://onnx.ai/onnx/operators/onnx__Reshape.html
        try compute_reshape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Resize")) {
        try compute_resize_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Shape")) {
        try compute_shape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sigmoid")) {
        try compute_sigmoid_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Softmax")) {
        //https://onnx.ai/onnx/operators/onnx__Softmax.html
        try compute_softmax_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Slice")) {
        try compute_slice_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Split")) {
        //https://onnx.ai/onnx/operators/onnx__Split.html
        try compute_split_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sub")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Transpose")) {
        try compute_transpose_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Unsqueeze")) {
        try compute_unsqueeze_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Identity")) {
        //https://onnx.ai/onnx/operators/onnx__Identity.html
        try compute_identity_output_shape(readyNode);
    } else {
        std.debug.print("\n\n ERROR! output shape computation for {s} is not available in codeGen_math_handler.compute_output_shape() \n\n", .{readyNode.nodeProto.op_type});
        return error.OperationNotSupported;
    }
}

// ---------------- SHAPE COMPUTATION METHODS ----------------
inline fn compute_Add_output_shape(readyNode: *ReadyNode) !void {
    var shape: []const i64 = undefined;

    if (utils.getTensorShape(readyNode.outputs.items[0].name)) |tensorShape| {
        shape = tensorShape;
    } else {
        shape = readyNode.inputs.items[0].shape;
    }
    readyNode.outputs.items[0].shape = shape;
}

inline fn compute_constant_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_constant_output_shape node: {s}======", .{readyNode.nodeProto.name.?});

    // Check each possible attribute type for the Constant node
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "value")) {
            // Handle tensor value - use existing utility
            const shape = try utils.getConstantTensorDims(readyNode.nodeProto);

            // If the shape is empty (scalar in ONNX), use [1] instead
            if (shape.len == 0) {
                readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
            } else {
                readyNode.outputs.items[0].shape = shape;
            }

            std.debug.print("\n output_shape from tensor: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "value_float") or std.mem.eql(u8, attr.name, "value_int") or
            std.mem.eql(u8, attr.name, "value_string"))
        {
            // These are scalar values - output shape is [1]
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
            std.debug.print("\n output_shape scalar: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "value_floats") or std.mem.eql(u8, attr.name, "value_ints")) {
            // These are 1D arrays - shape is [length]
            var length: i64 = 0;
            if (attr.type == AttributeType.FLOATS) {
                length = @intCast(attr.floats.len);
            } else if (attr.type == AttributeType.INTS) {
                length = @intCast(attr.ints.len);
            }
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{length});
            std.debug.print("\n output_shape 1D array: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "value_strings")) {
            // 1D array of strings - shape is [length]
            const length: i64 = @intCast(attr.strings.len);
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{length});
            std.debug.print("\n output_shape string array: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "sparse_value")) {
            // For sparse tensor, we need to handle it differently
            std.debug.print("\n Warning: Sparse tensor support is limited", .{});

            // Use a placeholder shape for sparse tensors - assuming scalar for now
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
            std.debug.print("\n output_shape from sparse tensor (placeholder): []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        }
    }

    return error.ConstantValueNotFound;
}

inline fn compute_ReLU_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_ReLU_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_reshape_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_reshape_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});

    // Check if the second input has a tensorProto with int64_data
    if (readyNode.inputs.items[1].tensorProto != null and
        readyNode.inputs.items[1].tensorProto.?.int64_data != null)
    {
        std.debug.print("\n new shape from tensorProto: []i64 = {any}", .{readyNode.inputs.items[1].tensorProto.?.int64_data.?});
        readyNode.outputs.items[0].shape = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    } else {
        // If not, use the shape of the second input directly
        std.debug.print("\n new shape from input shape: []i64 = {any}", .{readyNode.inputs.items[1].shape});
        readyNode.outputs.items[0].shape = try allocator.dupe(i64, readyNode.inputs.items[1].shape);
    }

    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
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

fn compute_mul_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_mul_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_a = readyNode.inputs.items[0];
    const input_b = readyNode.inputs.items[1];

    std.debug.print("\n input_a_shape: []i64 = {any}", .{input_a.shape});
    std.debug.print("\n input_b_shape: []i64 = {any}", .{input_b.shape});

    // For element-wise multiplication, shapes must be exactly the same
    if (input_a.shape.len != input_b.shape.len) {
        return error.MismatchedRank;
    }

    for (input_a.shape, input_b.shape) |a, b| {
        if (a != b) {
            return error.MismatchedShape;
        }
    }

    // Output shape is the same as input shapes
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, input_a.shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_conv_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_conv_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;
    const kernel_shape: []const i64 = readyNode.inputs.items[1].shape;

    var stride: ?[]i64 = null;
    var dilation: ?[]i64 = null;
    var auto_pad: []const u8 = "NOTSET";
    var pads: ?[]i64 = null;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "strides")) {
            if (attr.type == AttributeType.INTS) stride = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "dilations")) {
            if (attr.type == AttributeType.INTS) dilation = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "auto_pad")) {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s;
        }
        if (std.mem.eql(u8, attr.name, "pads")) {
            if (attr.type == AttributeType.INTS) pads = attr.ints;
        }
    }

    if (stride == null) return error.StridesNotFound;
    if (dilation == null) return error.DilationsNotFound;

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape});
    std.debug.print("\n stride: []i64 = {any}", .{stride.?});
    //std.debug.print("\n pads: []i64 = {any}", .{pads.?});
    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_convolution_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                try utils.i64SliceToUsizeSlice(kernel_shape),
                try utils.i64SliceToUsizeSlice(stride.?),
                if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
                try utils.i64SliceToUsizeSlice(dilation.?),
                auto_pad,
            ),
        ),
    );
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_maxPool_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_maxPool_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;

    var kernel_shape: ?[]i64 = null;
    var stride: ?[]i64 = null;
    var dilation: ?[]i64 = null;
    var auto_pad: []const u8 = "NOTSET";
    var pads: ?[]i64 = null;
    var ceil_mode: bool = false;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "kernel_shape")) {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "strides")) {
            if (attr.type == AttributeType.INTS) stride = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "dilations")) {
            if (attr.type == AttributeType.INTS) dilation = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "auto_pad")) {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s;
        } else if (std.mem.eql(u8, attr.name, "pads")) {
            if (attr.type == AttributeType.INTS) pads = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "ceil_mode")) {
            if (attr.type == AttributeType.INT) ceil_mode = attr.i != 0;
        }
    }

    if (kernel_shape == null) return error.KernelShapeNotFound;
    if (stride == null) return error.StridesNotFound;

    // Create proper allocated slices for default values
    var default_dilation: []i64 = undefined;
    var default_pads: []i64 = undefined;
    var should_free_dilation = false;
    var should_free_pads = false;

    if (dilation == null) {
        default_dilation = try allocator.alloc(i64, 2);
        default_dilation[0] = 1;
        default_dilation[1] = 1;
        dilation = default_dilation;
        should_free_dilation = true;
    }

    if (pads == null) {
        default_pads = try allocator.alloc(i64, 4);
        @memset(default_pads, 0);
        pads = default_pads;
        should_free_pads = true;
    }

    defer {
        if (should_free_dilation) allocator.free(default_dilation);
        if (should_free_pads) allocator.free(default_pads);
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape.?});
    std.debug.print("\n stride: []i64 = {any}", .{stride.?});
    std.debug.print("\n dilation: []i64 = {any}", .{dilation.?});
    std.debug.print("\n pads: []i64 = {any}", .{pads.?});
    std.debug.print("\n auto_pad: {s}", .{auto_pad});
    std.debug.print("\n ceil_mode: {}", .{ceil_mode});

    // Convert AutoPadType from string
    var auto_pad_type: tensorMath.AutoPadType = .NOTSET;
    if (std.mem.eql(u8, auto_pad, "VALID")) {
        auto_pad_type = .VALID;
    } else if (std.mem.eql(u8, auto_pad, "SAME_UPPER")) {
        auto_pad_type = .SAME_UPPER;
    } else if (std.mem.eql(u8, auto_pad, "SAME_LOWER")) {
        auto_pad_type = .SAME_LOWER;
    }

    // Convert parameters to usize
    const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
    defer allocator.free(usize_input_shape);

    const usize_kernel_shape = try utils.i64SliceToUsizeSlice(kernel_shape.?);
    defer allocator.free(usize_kernel_shape);

    const usize_stride = try utils.i64SliceToUsizeSlice(stride.?);
    defer allocator.free(usize_stride);

    const usize_dilation = try utils.i64SliceToUsizeSlice(dilation.?);
    defer allocator.free(usize_dilation);

    const usize_pads = try utils.i64SliceToUsizeSlice(pads.?);
    defer allocator.free(usize_pads);

    // Call the more complete maxpool shape function
    const output_shape = try tensorMath.get_onnx_maxpool_output_shape(usize_input_shape, usize_kernel_shape, usize_stride, usize_dilation, usize_pads, auto_pad_type, ceil_mode);
    defer allocator.free(output_shape);

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_reducemean_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_reducemean_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = try utils.i64SliceToUsizeSlice(readyNode.inputs.items[0].shape);
    defer allocator.free(input_shape);

    // Get attributes
    var keepdims: bool = true;
    var noop_with_empty_axes: bool = false;
    var axes_attr: ?[]i64 = null;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "keepdims")) {
            if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
            if (attr.type == AttributeType.INT) noop_with_empty_axes = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "axes")) {
            if (attr.type == AttributeType.INTS) axes_attr = attr.ints;
        }
    }

    // Get axes from second input if it exists
    var axes: ?[]const i64 = null;
    if (readyNode.inputs.items.len > 1 and
        readyNode.inputs.items[1].tensorProto != null and
        readyNode.inputs.items[1].tensorProto.?.int64_data != null)
    {
        axes = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    } else if (axes_attr != null) {
        // Use axes from attribute if not found in inputs
        axes = axes_attr;
    }

    std.debug.print("\n input_shape: []usize = {any}", .{input_shape});
    std.debug.print("\n axes: ?[]i64 = {any}", .{axes});
    std.debug.print("\n keepdims: {}", .{keepdims});
    std.debug.print("\n noop_with_empty_axes: {}", .{noop_with_empty_axes});

    const output_shape = try tensorMath.get_reduce_mean_output_shape(input_shape, axes, keepdims, noop_with_empty_axes);
    defer allocator.free(output_shape);

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_slice_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_slice_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    const starts = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    const ends = readyNode.inputs.items[2].tensorProto.?.int64_data.?;

    var axes: ?[]i64 = null;
    var steps: ?[]i64 = null;

    // Get axes if provided (input 3)
    if (readyNode.inputs.items.len > 3) {
        axes = readyNode.inputs.items[3].tensorProto.?.int64_data.?;
    }

    // Get steps if provided (input 4)
    if (readyNode.inputs.items.len > 4) {
        steps = readyNode.inputs.items[4].tensorProto.?.int64_data.?;
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n starts: []i64 = {any}", .{starts});
    std.debug.print("\n ends: []i64 = {any}", .{ends});
    std.debug.print("\n axes: []i64 = {any}", .{axes});
    std.debug.print("\n steps: []i64 = {any}", .{steps});

    const output_shape = try tensorMath.get_slice_output_shape(
        try utils.i64SliceToUsizeSlice(input_shape),
        starts,
        ends,
        axes,
        steps,
    );

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_shape_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_shape_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;

    // Get start and end attributes if they exist
    var start: ?i64 = null;
    var end: ?i64 = null;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "start")) {
            if (attr.type == AttributeType.INT) start = attr.i;
        } else if (std.mem.eql(u8, attr.name, "end")) {
            if (attr.type == AttributeType.INT) end = attr.i;
        }
    }

    // Calculate output size
    const rank = input_shape.len;
    var start_axis: i64 = start orelse 0;
    if (start_axis < 0) start_axis += @as(i64, @intCast(rank));
    start_axis = @max(0, @min(start_axis, @as(i64, @intCast(rank - 1))));

    var end_axis: i64 = end orelse @as(i64, @intCast(rank));
    if (end_axis < 0) end_axis += @as(i64, @intCast(rank));
    end_axis = @max(start_axis, @min(end_axis, @as(i64, @intCast(rank))));

    const output_size = @max(0, end_axis - start_axis);

    // Shape operator always outputs a 1D tensor
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{@intCast(output_size)});
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_gather_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_gather_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const data_shape = readyNode.inputs.items[0].shape;
    const indices_shape = readyNode.inputs.items[1].shape;

    // Get axis attribute, default is 0
    var axis: i64 = 0;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        }
    }

    // Handle negative axis
    if (axis < 0) {
        axis += @as(i64, @intCast(data_shape.len));
    }

    std.debug.print("\n data_shape: []i64 = {any}", .{data_shape});
    std.debug.print("\n indices_shape: []i64 = {any}", .{indices_shape});
    std.debug.print("\n axis: {}", .{axis});

    // Calculate output shape:
    // - Take all dimensions from data before axis
    // - Add all dimensions from indices
    // - Take all dimensions from data after axis
    const output_rank = data_shape.len + indices_shape.len - 1;
    var output_shape = try allocator.alloc(i64, output_rank);

    // Copy dimensions from data before axis
    var out_idx: usize = 0;
    for (0..@as(usize, @intCast(axis))) |i| {
        output_shape[out_idx] = data_shape[i];
        out_idx += 1;
    }

    // Copy dimensions from indices
    for (indices_shape) |dim| {
        output_shape[out_idx] = dim;
        out_idx += 1;
    }

    // Copy remaining dimensions from data after axis
    for (@as(usize, @intCast(axis + 1))..data_shape.len) |i| {
        output_shape[out_idx] = data_shape[i];
        out_idx += 1;
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_sigmoid_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_sigmoid_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Sigmoid is element-wise, output shape is identical to input shape
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, input_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_transpose_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_transpose_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;

    // Get the perm attribute if it exists
    var perm: ?[]i64 = null;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "perm")) {
            if (attr.type == AttributeType.INTS) {
                perm = attr.ints;
            }
        }
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n perm: []i64 = {any}", .{perm});

    // If no perm is provided, reverse the dimensions
    var output_shape = try allocator.alloc(i64, input_shape.len);
    if (perm) |p| {
        // Validate perm length
        if (p.len != input_shape.len) {
            return error.InvalidPermutation;
        }
        // Apply permutation
        for (p, 0..) |axis, i| {
            if (axis < 0 or axis >= @as(i64, @intCast(input_shape.len))) {
                return error.InvalidPermutation;
            }
            output_shape[i] = input_shape[@intCast(axis)];
        }
    } else {
        // Reverse dimensions
        for (input_shape, 0..) |_, i| {
            output_shape[i] = input_shape[input_shape.len - 1 - i];
        }
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_unsqueeze_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_unsqueeze_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Get axes from attributes or from the second input tensor
    var axes: ?[]const i64 = null;

    // First check if axes is provided as an input tensor (ONNX opset 13+)
    if (readyNode.inputs.items.len > 1 and readyNode.inputs.items[1].tensorProto != null) {
        axes = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
        std.debug.print("\n axes from input tensor: []i64 = {any}", .{axes.?});
    } else {
        // Otherwise, check for axes attribute (ONNX opset < 13)
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axes")) {
                if (attr.type == AttributeType.INTS) {
                    axes = attr.ints;
                    std.debug.print("\n axes from attribute: []i64 = {any}", .{axes.?});
                    break;
                }
            }
        }
    }

    if (axes == null) return error.UnsqueezeAxesNotFound;

    // Calculate output shape by inserting dimensions at the specified axes
    const output_rank = input_shape.len + axes.?.len;
    var output_shape = try allocator.alloc(i64, output_rank);

    // Initialize with 1s
    @memset(output_shape, 1);

    // Create a mask to track which positions are for the new dimensions
    var is_unsqueezed_axis = try allocator.alloc(bool, output_rank);
    defer allocator.free(is_unsqueezed_axis);
    @memset(is_unsqueezed_axis, false);

    // Mark the positions where dimensions will be inserted
    for (axes.?) |axis| {
        var normalized_axis = axis;
        if (normalized_axis < 0) normalized_axis += @as(i64, @intCast(output_rank));

        if (normalized_axis < 0 or normalized_axis >= @as(i64, @intCast(output_rank))) {
            return error.UnsqueezeInvalidAxis;
        }

        is_unsqueezed_axis[@intCast(normalized_axis)] = true;
    }

    // Fill in the output shape with the input dimensions
    var input_idx: usize = 0;
    for (0..output_rank) |output_idx| {
        if (!is_unsqueezed_axis[output_idx]) {
            output_shape[output_idx] = input_shape[input_idx];
            input_idx += 1;
        }
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

pub fn compute_concat_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n compute_concat_output_shape for node: {s}", .{readyNode.nodeProto.name.?});

    // Get the axis attribute (required)
    var axis: i64 = 0;
    var axis_found = false;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) {
                axis = attr.i;
                axis_found = true;
            } else {
                return error.ConcatAxisNotINT;
            }
        }
    }

    if (!axis_found) {
        return error.ConcatAxisNotFound;
    }

    std.debug.print("\n   axis: {}", .{axis});

    // Ensure there's at least one input tensor
    if (readyNode.inputs.items.len == 0) {
        return error.ConcatNoInputs;
    }

    // Convert input shapes to usize for get_concatenate_output_shape
    var input_shapes = try allocator.alloc([]const usize, readyNode.inputs.items.len);
    defer {
        for (input_shapes) |shape| {
            allocator.free(shape);
        }
        allocator.free(input_shapes);
    }

    for (readyNode.inputs.items, 0..) |input, i| {
        input_shapes[i] = try utils.i64SliceToUsizeSlice(input.shape);
    }

    // Get output shape using the existing function
    const output_shape = try tensorMath.get_concatenate_output_shape(input_shapes, @intCast(axis));
    defer allocator.free(output_shape);

    // Convert back to i64 for storing in readyNode
    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n   output shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_ceil_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_ceil_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Ceil is an element-wise operation, output shape is identical to input shape
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, input_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_identity_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_identity_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Identity operation preserves the input shape
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, input_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_leaky_relu_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_leaky_relu_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // LeakyReLU is an element-wise operation, output shape is identical to input shape
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, input_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

fn compute_matmul_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_matmul_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_a = readyNode.inputs.items[0];
    const input_b = readyNode.inputs.items[1];

    std.debug.print("\n input_a_shape: []i64 = {any}", .{input_a.shape});
    std.debug.print("\n input_b_shape: []i64 = {any}", .{input_b.shape});

    if (input_a.shape.len < 2 or input_b.shape.len < 2) {
        return error.InvalidShape;
    }

    const a_rows = input_a.shape[input_a.shape.len - 2];
    const a_cols = input_a.shape[input_a.shape.len - 1];
    const b_rows = input_b.shape[input_b.shape.len - 2];
    const b_cols = input_b.shape[input_b.shape.len - 1];

    if (a_cols != b_rows) {
        return error.ShapeMismatch;
    }

    var output_shape = try allocator.alloc(i64, input_a.shape.len);
    errdefer allocator.free(output_shape);

    // Copy batch dimensions from input_a
    for (0..input_a.shape.len - 2) |i| {
        output_shape[i] = input_a.shape[i];
    }

    // Set matrix multiplication dimensions
    output_shape[output_shape.len - 2] = a_rows;
    output_shape[output_shape.len - 1] = b_cols;

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_split_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_split_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;

    // Get axis attribute (default is 0)
    var axis: i64 = 0;
    var split_sizes: ?[]i64 = null;

    // Extract attributes
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        } else if (std.mem.eql(u8, attr.name, "split")) {
            if (attr.type == AttributeType.INTS) split_sizes = attr.ints;
        }
    }

    // Check if split_sizes is provided as an input (ONNX opset 13+)
    if (readyNode.inputs.items.len > 1 and
        readyNode.inputs.items[1].tensorProto != null and
        readyNode.inputs.items[1].tensorProto.?.int64_data != null)
    {
        split_sizes = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n axis: {}", .{axis});
    std.debug.print("\n split_sizes: {any}", .{split_sizes});
    std.debug.print("\n num_outputs: {}", .{readyNode.outputs.items.len});

    // Convert i64 split_sizes to usize if provided
    var usize_split_sizes: ?[]usize = null;
    defer if (usize_split_sizes != null) allocator.free(usize_split_sizes.?);

    if (split_sizes) |sizes| {
        usize_split_sizes = try allocator.alloc(usize, sizes.len);
        for (sizes, 0..) |size, i| {
            usize_split_sizes.?[i] = @intCast(size);
        }
    }

    // Convert input_shape to usize
    const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
    defer allocator.free(usize_input_shape);

    // Get output shapes using the utility function
    const output_shapes = try tensorMath.get_split_output_shapes(usize_input_shape, axis, usize_split_sizes, readyNode.outputs.items.len // Pass the number of outputs
    );
    defer {
        for (output_shapes) |shape| {
            allocator.free(shape);
        }
        allocator.free(output_shapes);
    }

    // Ensure we have enough output tensors
    if (readyNode.outputs.items.len != output_shapes.len) {
        return error.MismatchedOutputCount;
    }

    // Set the output shapes
    for (output_shapes, 0..) |shape, i| {
        readyNode.outputs.items[i].shape = try utils.usizeSliceToI64Slice(shape);
        std.debug.print("\n output[{}] shape: []i64 = {any}", .{ i, readyNode.outputs.items[i].shape });
    }
}

pub fn compute_resize_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_resize_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Extract scales and sizes from inputs
    var scales: ?[]const f32 = null;
    var sizes: ?[]const i64 = null;

    std.debug.print("\n KKKKKKKKKKK {}", .{readyNode.inputs.items.len});
    for (readyNode.inputs.items) |i| {
        i.print(true);
    }
    // ROI is at index 1, scales at index 2, sizes at index 3
    if (readyNode.inputs.items.len > 2 and readyNode.inputs.items[2].tensorProto != null) {
        std.debug.print("\n AAAAAAAAAA", .{});
        if (readyNode.inputs.items[2].tensorProto.?.float_data != null) {
            std.debug.print("\n aaaaaaaaaaaaaaaa", .{});

            scales = readyNode.inputs.items[2].tensorProto.?.float_data.?;
            std.debug.print("\n scales: []f32 = {any}", .{scales});
        }
    }
    if (readyNode.inputs.items.len > 3 and readyNode.inputs.items[3].tensorProto != null) {
        std.debug.print("\n BBBBBBBBBBB", .{});

        if (readyNode.inputs.items[3].tensorProto.?.int64_data != null) {
            std.debug.print("\n bbbbbbbbbbbbbbbbbb", .{});

            sizes = readyNode.inputs.items[3].tensorProto.?.int64_data.?;
            std.debug.print("\n sizes: []i64 = {any}", .{sizes});
        }
    }

    // Convert input_shape to usize for the computation function
    const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
    defer allocator.free(usize_input_shape);

    // Convert sizes to usize if present --------> TODO why not using utils.sliceToUsizeSlice() ??
    var usize_sizes: ?[]const usize = null;
    var sizes_buffer: []usize = undefined;
    defer if (usize_sizes != null) allocator.free(sizes_buffer);

    if (sizes) |sz| {
        sizes_buffer = try allocator.alloc(usize, sz.len);
        for (sz, 0..) |s, i| {
            sizes_buffer[i] = @intCast(s);
        }
        usize_sizes = sizes_buffer;
    }

    // Calculate output shape using existing function
    const output_shape = try tensorMath.get_resize_output_shape(usize_input_shape, scales, usize_sizes);
    defer allocator.free(output_shape);

    // Convert back to i64 for storing in readyNode
    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

pub fn compute_resize_output_shape_generic(comptime T: type, input_shape: []const T, scales: ?[]const f32, sizes: ?[]const T) ![]T {
    // Make sure we support all parameter types
    if (scales != null) {
        // Calculate output shape based on scales
        var output_shape = try allocator.alloc(T, input_shape.len);

        for (0..input_shape.len) |i| {
            output_shape[i] = @intFromFloat(@as(f32, @floatFromInt(input_shape[i])) * scales.?[i]);
        }

        return output_shape;
    }

    if (sizes != null) {
        // Use sizes directly
        return sizes.?;
    }

    // If neither scales nor sizes is provided, return the input shape
    return input_shape;
}

inline fn compute_neg_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_neg_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Neg operation preserves the input shape - use the utility function
    const usize_input_shape = try utils.i64SliceToUsizeSlice(input_shape);
    defer allocator.free(usize_input_shape);

    const output_shape = try tensorMath.get_neg_output_shape(usize_input_shape);
    defer allocator.free(output_shape);

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}
