const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeProto = onnx.AttributeProto;
const allocator = zant.utils.allocator.allocator;

const codegen = @import("codegen").codegen_v2;
const utils = codegen.utils;
const codegen_options = @import("codegen_options");

const globals = codegen.globals;
const ReadyNode = globals.ReadyNode;
const ReadyTensor = globals.ReadyTensor;
const NodeZant = zant.IR_graph.NodeZant;
const Builder = zant.uops.UOpBuilder;
const ZigRenderer = codegen.renderer.ZigRenderer;

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: *std.Io.Writer, nodes: std.ArrayList(*NodeZant), do_export: bool) !void {
    // Static initialization for output tensors if not using dynamic allocation
    if (!codegen_options.dynamic) {
        // declare all the outputs of each node of the network
        try write_outputsInitialization(writer);
        // method to reset the tensors values
        try write_outputsResetMethod(writer);
    }

    _ = try writer.print(
        \\
        \\
        \\
        \\pub {s} fn predict( 
        \\    input: [*]T,
        \\    input_shape: [*]u32,
        \\    shape_len: u32,
        \\    result: *[*]T,
        \\) void {{
    , .{if (do_export == true) "export" else ""});

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Starting prediction...\n")));
            \\    }}
        , .{});
    }

    if (!codegen_options.dynamic) {
        _ = try writer.print(
            \\
            \\    // Reset all output tensors to zero before each prediction
            \\    resetOutputTensors();
        , .{});
    }

    try write_checks(writer);

    try write_predictInitialization(writer);

    try write_graphSerialization(writer, nodes);

    try writeReturn(writer);

    _ = try writer.print(
        \\
        \\}} 
    , .{});
}

// Processes and writes the computation graph
inline fn write_graphSerialization(writer: *std.Io.Writer, nodes: std.ArrayList(*NodeZant)) !void {
    const Writer = @TypeOf(writer);

    // initializing the renderer and the builder
    var renderer = ZigRenderer(Writer).init(zant.utils.allocator.allocator, writer);

    var builder = Builder.init(zant.utils.allocator.allocator);
    defer builder.deinit();

    for (nodes.items) |node| {
        try node.render_lower_math_op(&builder);
    }

    const uop_list = try builder.toOwnedSlice();

    switch (nodes.items[0].op) {
        .reshape => |op| {
            try renderer.render_body(uop_list, &.{op.reshaped.get_tensorZantID()});
        },
        else => {},
    }
}

// -------------------------------- WRITE OUTPUTS --------------------------------

// Initializes output tensors in the computation graph
fn write_outputsInitialization(writer: *std.Io.Writer) !void {
    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         Initializing output Tensors             +
        \\ // ---------------------------------------------------
    , .{});

    for (globals.readyGraph.items) |*node| {

        //writing the outputs, OSS: two nodes shpuld never have the same output by definition, so we don't need to check for duplicates
        for (node.outputs.items) |output| {
            if (std.mem.eql(u8, node.nodeProto.op_type, "Constant") and node.inputs.items.len == 0) { //A node is constant if it only has one output and no inputs
                if (node.outputs.items.len > 1) return error.MultipleOutputConstant else {
                    try write_constantTensor(writer, node);
                    //set the node and tensor to Ready
                    var mutableNode: *ReadyNode = @constCast(node);
                    mutableNode.ready = true;
                }
            } else {
                if (@as(?*ReadyTensor, output) == null) return error.InvalidOutput;
                const size = try write_OutputShape(
                    writer,
                    output,
                    node,
                );
                try write_OutputTensor(
                    writer,
                    output,
                    size,
                );
            }
        }
    }
}

fn write_OutputShape(writer: *std.Io.Writer, output: *ReadyTensor, node: *const ReadyNode) !i64 {
    if (@as(?*ReadyTensor, output) == null) return error.InvalidOutput;
    const original_shape = output.shape;
    var size: i64 = 1;

    // Check if it's a convolutional node
    const op_type = node.nodeProto.op_type;
    const is_conv = std.mem.eql(u8, op_type, "Conv") or std.mem.eql(u8, op_type, "ConvInteger");
    const is_cast = std.mem.eql(u8, op_type, "Cast"); // Check for Cast node
    const is_add = std.mem.eql(u8, op_type, "Add"); // Check for Add node

    var shape_len_adj: usize = original_shape.len;
    var needs_batch_dim: bool = false;

    // Determine if a batch dimension needs to be added
    if (is_conv and (original_shape.len == 0 or original_shape[0] != 1)) {
        needs_batch_dim = true;
        shape_len_adj += 1;
    } else if (is_conv and original_shape.len > 0 and original_shape[0] == 1) {
        // Already has batch dim 1, no change needed for conv
    } else if (is_cast and original_shape.len == 3) { // Add check for Cast with 3 dims
        needs_batch_dim = true;
        shape_len_adj += 1;
    } else if (is_add and original_shape.len == 3) { // Add check for Add with 3 dims
        needs_batch_dim = true;
        shape_len_adj += 1;
    } else {
        // Not a conv/cast/add node needing adjustment, or already has batch dim
    }

    try writer.print(
        \\
        \\
        \\var shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{
        try utils.getSanitizedName(output.name),
        shape_len_adj, // Use adjusted length
    });

    var first_dim_written = false;
    if (needs_batch_dim) {
        try writer.print(" 1", .{}); // Add batch dimension of 1
        size *= 1;
        first_dim_written = true;
    }

    for (0..original_shape.len) |i| {
        if (first_dim_written or i > 0) try writer.print(",", .{});
        try writer.print(
            \\ {}
        , .{original_shape[i]});
        size *= original_shape[i];
        first_dim_written = true; // Ensure comma is added after the first element (batch or original[0])
    }

    try writer.print(
        \\}} ;
    , .{});

    return size;
}

fn write_constantTensor(writer: *std.Io.Writer, readyNode: *const ReadyNode) !void {
    try writer.print(
        \\
        \\ // ---- CONSTANT TENSOR ---- 
    , .{});

    // Get the output tensor (constant nodes have exactly one output)
    const output = readyNode.outputs.items[0];
    const sanitized_name = try utils.getSanitizedName(output.name);

    // Find the value attribute which contains the constant tensor
    var value_attr: ?*AttributeProto = null;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "value")) {
            value_attr = attr;
            break;
        }
    }

    if (value_attr == null or value_attr.?.t == null) return error.MissingConstantValue;
    const tensor = value_attr.?.t.?;

    // Write shape array
    try writer.print(
        \\
        \\const shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{ sanitized_name, output.shape.len });

    for (0..output.shape.len) |i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print(
            \\ {}
        , .{output.shape[i]});
    }

    try writer.print(
        \\}} ;
    , .{});

    // Write data array
    var total_size: i64 = 1;
    for (tensor.dims) |dim| {
        total_size *= dim;
    }

    //const dataTypeString = try utils.getTypeString(tensor.data_type);
    const type_str_const = try utils.getTypeString(tensor.data_type);
    try writer.print(
        \\
        \\const array_{s} : [{d}]{s} = [_]{s}{{
    , .{ sanitized_name, total_size, type_str_const, type_str_const });

    // Write the actual data values
    if (tensor.float_data) |data| {
        for (0..data.len) |i| {
            if (i > 0) try writer.print(",", .{});
            try writer.print(" {d}", .{data[i]});
        }
    } else if (tensor.int64_data) |data| {
        for (0..data.len) |i| {
            if (i > 0) try writer.print(",", .{});
            try writer.print(" {d}", .{data[i]});
        }
    } else if (tensor.raw_data) |data| {
        switch (tensor.data_type) {
            .FLOAT => {
                const float_data = @as([*]const f32, @ptrCast(@alignCast(data.ptr)))[0..@divExact(data.len, 4)];
                for (0..float_data.len) |i| {
                    if (i > 0) try writer.print(",", .{});
                    try writer.print(" {d}", .{float_data[i]});
                }
            },
            .INT64 => {
                const int_data = @as([*]const i64, @ptrCast(@alignCast(data.ptr)))[0..@divExact(data.len, 8)];
                for (0..int_data.len) |i| {
                    if (i > 0) try writer.print(",", .{});
                    try writer.print(" {d}", .{int_data[i]});
                }
            },
            else => return error.UnsupportedDataType,
        }
    } else return error.NoDataAvailable;

    try writer.print(
        \\ }};
    , .{});

    // Write tensor initialization using fromArray
    try writer.print(
        \\
        \\const tensor_{s} = Tensor({s}).fromConstBuffer(&allocator, &array_{s}, &shape_tensor_{s});
    , .{ sanitized_name, type_str_const, sanitized_name, sanitized_name });
}

fn write_OutputTensor(writer: *std.Io.Writer, output: *ReadyTensor, size: i64) !void {
    const sanitized_name = try utils.getSanitizedName(output.name);

    // --- ADD CHECK FOR UNDEFINED TYPE ---
    if (output.dtype == .UNDEFINED) {
        std.log.warn("\n\nCODEGEN ERROR: Attempted to generate output tensor '{s}' but its data type is UNDEFINED. Check ONNX graph analysis in globals.zig.\n\n", .{output.name});
        return error.DataTypeNotAvailable; // Or a more specific error like CannotGenerateUndefinedType
    }
    // --- END CHECK ---

    const type_str = try utils.getTypeString(output.dtype);
    if (codegen_options.dynamic) {
        // Check if this is the final network output tensor
        if (std.mem.eql(u8, output.name, globals.networkOutput.name)) {
            // Network Output: Allocate but DO NOT defer free/deinit. Caller takes ownership.
            _ = try writer.print(
                \\
                \\    // Allocate final network output buffer (caller owns this memory)
                \\    var array_{s} = allocator.alloc({s}, {d}) catch return;
                \\    var tensor_{s} = Tensor({s}).fromArray(&allocator, array_{s}, &shape_tensor_{s});
                \\    // NOTE: No 'defer allocator.free(array_{s})' or 'defer tensor_{s}.deinit()'
                \\    //       The pointer returned by predict() must be freed by the caller.
            , .{ sanitized_name, type_str, size, sanitized_name, type_str, sanitized_name });
        } else {
            // Intermediate Tensor: Allocate AND defer free/deinit.
            const code_str = try std.fmt.allocPrint(allocator,
                \\    var array_{s} = allocator.alloc({s}, {d}) catch return;
                \\    defer allocator.free(array_{s}); // Free intermediate array
                \\    var tensor_{s} = Tensor({s}).fromArray(&allocator, array_{s}, &shape_tensor_{s});
                \\    defer tensor_{s}.deinit(); // Deinit intermediate tensor struct
            , .{ sanitized_name, type_str, size, sanitized_name, type_str, sanitized_name });
            defer allocator.free(code_str);
            try writer.writeAll(code_str);
        }
    } else {
        // Static allocation: Use fromConstBuffer to allow mutation
        try writer.print("    var array_{s}: [{d}]{s} = [_]{s}{{0}} ** {d};", .{ sanitized_name, size, type_str, type_str, size });
        try writer.print("    var tensor_{s} = Tensor({s}).fromConstBuffer(&fba, &array_{s}, &shape_tensor_{s});", .{ sanitized_name, type_str, sanitized_name, sanitized_name });
    }
}

fn write_outputsResetMethod(writer: *std.Io.Writer) !void {
    try writer.print(
        \\
        \\
        \\//Function to reset all output tensors to zero
        \\fn resetOutputTensors() void {{
    , .{});

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Resetting output tensors...\n")));
            \\    }}
        , .{});
    }

    for (globals.readyGraph.items) |*node| {
        // Skip constant nodes
        if (std.mem.eql(u8, node.nodeProto.op_type, "Constant") and node.inputs.items.len == 0) {
            continue;
        }

        for (node.outputs.items) |output| {
            _ = try writer.print(
                \\
                \\    @memset(array_{s}[0..], 0);
            , .{try utils.getSanitizedName(output.name)});
        }
    }

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Output tensors reset.\n")));
            \\    }}
        , .{});
    }

    try writer.print(
        \\
        \\}}
    , .{});
}

// -------------------------------- WRITE CHECKS --------------------------------

fn write_checks(writer: *std.Io.Writer) !void {
    // Autogen a check for the input shape as arg VS input shape as codegen option

    //check on the number of dims
    _ = try writer.print(
        \\
        \\    //checks on the input parameters
        \\    if (shape_len == 0) return ;
        \\    if(shape_len != {}) return ;
    , .{globals.networkInput.shape.len});

    //check on dims correspondance
    for (globals.networkInput.shape, 0..) |dim, i| {
        _ = try writer.print(
            \\
            \\    if( input_shape[{}] != {}) return ;
        , .{ i, dim });
    }
}

// -------------------------------- WRITE PREDICT() --------------------------------

fn write_predictInitialization(writer: *std.Io.Writer) !void {
    _ = try writer.print(
        \\  
        \\    //computing the size of the input tensor
        \\    var size: u32 = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        size *= input_shape[dim_i];
        \\    }}
        \\     
        \\    //allocating space in memory for the data
        \\    const data = allocator.alloc(T, size) catch return;
        \\    defer allocator.free(data);
        \\    for (0..size) |i| {{
        \\        data[i] = input[i]; // Copying input elements 
        \\    }}
    , .{});
}

fn writeReturn(writer: *std.Io.Writer) !void {
    _ = try writer.print(
        \\
        \\    result.* = tensor_{s}.data.ptr;
        \\
    , .{try utils.getSanitizedName(globals.networkOutput.name)});

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Prediction completed.\n")));
            \\    }}
        , .{});
    }
}
