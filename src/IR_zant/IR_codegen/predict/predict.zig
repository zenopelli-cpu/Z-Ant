const std = @import("std");
const zant = @import("zant");

const IR_zant = @import("../../IR_zant.zig");
const IR_graph = IR_zant.IR_graph;
const IR_codegen = IR_zant.IR_codegen;

// --- zant IR
const IR_utils = IR_graph.utils;
const GraphZant = IR_graph.GraphZant;
const TensorZant = IR_graph.TensorZant;
const NodeZant = IR_graph.NodeZant;

const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_graph.tensorZant_lib.tensorMap;

const allocator = std.heap.page_allocator;

const codegen_options = @import("codegen_options");

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant), do_export: bool) !void {

    // Static initialization for output tensors if not using dynamic allocation
    //
    // declare all the outputs for each node, aka: linkers
    try write_linkersInitialization(writer);

    // declare all the outputs of  the network
    try write_outputsInitialization(writer);

    // method to reset the tensors values
    try write_linkersResetMethod(writer);

    const inputs = try IR_utils.getInputs(tensorZantMap);
    const outputs = try IR_utils.getInputs(tensorZantMap);

    //write input type
    _ = try writer.print(
        \\
        \\ const T_in : type = {s};
    , .{inputs[0].ty.toString()});

    //write output type
    _ = try writer.print(
        \\
        \\ const T_out : type = {s};
    , .{outputs[0].ty.toString()});

    _ = try writer.print(
        \\
        \\
        \\
        \\pub {s} fn predict(
        \\    input: [*]T_in,
        \\    input_shape: [*]u32,
        \\    shape_len: u32,
        \\    result: *[*]T_out,
        \\) void {{
    , .{if (do_export == true) "export" else ""});

    if (codegen_options.IR_log) {
        _ = try writer.print(
            \\
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Starting prediction...\n")));
            \\    }}
        , .{});
    }

    _ = try writer.print(
        \\
        \\    // Reset all linker tensors to zero before each prediction
        \\    resetOutputTensors();
    , .{});

    try write_checks(writer);

    try write_predictInitialization(writer);

    try write_graphSerialization(writer, linearizedGraph);

    try writeReturn(writer);

    _ = try writer.print(
        \\
        \\}}
    , .{});
}

// -------------------------------- WRITE LINKERS --------------------------------

// Initializes output tensor of each node in the computation graph
fn write_linkersInitialization(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         Initializing linkers Tensors             +
        \\ // ---------------------------------------------------
    , .{});

    const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);

    for (linkers) |*tz| {
        const size = try write_TensorShape(
            writer,
            tz,
        );
        try write_TensorAllocation(
            writer,
            tz,
            size,
        );
    }
}

// fn write_constantTensor(writer: std.fs.File.Writer, readyNode: *const ReadyNode) !void {
//     try writer.print(
//         \\
//         \\ // ---- CONSTANT TENSOR ----
//     , .{});

//     // Get the output tensor (constant nodes have exactly one output)
//     const output = readyNode.outputs.items[0];
//     const sanitized_name = try utils.getSanitizedName(output.name);

//     // Find the value attribute which contains the constant tensor
//     var value_attr: ?*AttributeProto = null;
//     for (readyNode.nodeProto.attribute) |attr| {
//         if (std.mem.eql(u8, attr.name, "value")) {
//             value_attr = attr;
//             break;
//         }
//     }

//     if (value_attr == null or value_attr.?.t == null) return error.MissingConstantValue;
//     const tensor = value_attr.?.t.?;

//     // Write shape array
//     try writer.print(
//         \\
//         \\const shape_tensor_{s} : [{}]usize = [_]usize{{
//     , .{ sanitized_name, output.shape.len });

//     for (0..output.shape.len) |i| {
//         if (i > 0) try writer.print(",", .{});
//         try writer.print(
//             \\ {}
//         , .{output.shape[i]});
//     }

//     try writer.print(
//         \\}} ;
//     , .{});

//     // Write data array
//     var total_size: i64 = 1;
//     for (tensor.dims) |dim| {
//         total_size *= dim;
//     }

//     //const dataTypeString = try utils.getTypeString(tensor.data_type);
//     const type_str_const = try utils.getTypeString(tensor.data_type);
//     try writer.print(
//         \\
//         \\const array_{s} : [{d}]{s} = [_]{s}{{
//     , .{ sanitized_name, total_size, type_str_const, type_str_const });

//     // Write the actual data values
//     if (tensor.float_data) |data| {
//         for (0..data.len) |i| {
//             if (i > 0) try writer.print(",", .{});
//             try writer.print(" {d}", .{data[i]});
//         }
//     } else if (tensor.int64_data) |data| {
//         for (0..data.len) |i| {
//             if (i > 0) try writer.print(",", .{});
//             try writer.print(" {d}", .{data[i]});
//         }
//     } else if (tensor.raw_data) |data| {
//         switch (tensor.data_type) {
//             .FLOAT => {
//                 const float_data = @as([*]const f32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
//                 for (0..float_data.len) |i| {
//                     if (i > 0) try writer.print(",", .{});
//                     try writer.print(" {d}", .{float_data[i]});
//                 }
//             },
//             .INT64 => {
//                 const int_data = @as([*]const i64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
//                 for (0..int_data.len) |i| {
//                     if (i > 0) try writer.print(",", .{});
//                     try writer.print(" {d}", .{int_data[i]});
//                 }
//             },
//             else => return error.UnsupportedDataType,
//         }
//     } else return error.NoDataAvailable;

//     try writer.print(
//         \\ }};
//     , .{});

//     // Write tensor initialization using fromArray
//     try writer.print(
//         \\
//         \\const tensor_{s} = Tensor({s}).fromConstBuffer(&allocator, &array_{s}, &shape_tensor_{s});
//     , .{ sanitized_name, type_str_const, sanitized_name, sanitized_name });
// }

fn write_TensorAllocation(writer: std.fs.File.Writer, tz: *TensorZant, size: i64) !void {
    const sanitized_name = try tz.getNameSanitized();

    // --- ADD CHECK FOR UNDEFINED TYPE ---
    if (tz.ty == .undefined) {
        std.log.warn("\n\nCODEGEN ERROR: Attempted to generate output tensor '{s}' but its data type is UNDEFINED. Check ONNX graph analysis in globals.zig.\n\n", .{sanitized_name});
        return error.DataTypeNotAvailable; // Or a more specific error like CannotGenerateUndefinedType
    }
    // --- END CHECK ---

    const type_str = tz.ty.toString();

    // Static allocation: Use fromConstBuffer to allow mutation
    try writer.print("    var array_{s}: [{d}]{s} = [_]{s}{{0}} ** {d};", .{ sanitized_name, size, type_str, type_str, size });
    try writer.print("    var tensor_{s} = Tensor({s}).fromConstBuffer(&fba, &array_{s}, &shape_tensor_{s});", .{ sanitized_name, type_str, sanitized_name, sanitized_name });
}

fn write_linkersResetMethod(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\
        \\
        \\//Function to reset all output tensors to zero
        \\fn resetOutputTensors() void {{
    , .{});

    if (codegen_options.IR_log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Resetting output tensors...\n")));
            \\    }}
        , .{});
    }

    // --------- linkers
    const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);

    for (linkers) |*tz| {
        _ = try writer.print(
            \\
            \\    @memset(array_{s}[0..], 0);
        , .{try tz.getNameSanitized()});

        if (codegen_options.IR_log) {
            _ = try writer.print(
                \\
                \\    if (log_function) |log| {{
                \\        log(@constCast(@ptrCast("Linker tensor {s} reset.\n")));
                \\    }}
            , .{try tz.getNameSanitized()});
        }
    }

    // --------- outputs
    const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

    for (outputs) |*tz| {
        _ = try writer.print(
            \\
            \\    @memset(array_{s}[0..], 0);
        , .{try tz.getNameSanitized()});

        if (codegen_options.IR_log) {
            _ = try writer.print(
                \\
                \\    if (log_function) |log| {{
                \\        log(@constCast(@ptrCast("Output tensor {s} reset.\n")));
                \\    }}
            , .{try tz.getNameSanitized()});
        }
    }

    try writer.print(
        \\
        \\}}
    , .{});
}

// -------------------------------- WRITE OUTPUT --------------------------------
// Initializes output tensor of each node in the computation graph
fn write_outputsInitialization(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         Initializing Output Tensors             +
        \\ // ---------------------------------------------------
    , .{});

    const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

    for (outputs) |*tz| {
        const size = try write_TensorShape(
            writer,
            tz,
        );
        try write_TensorAllocation(
            writer,
            tz,
            size,
        );
    }
}

// -------------------------------- WRITE PREDICT() --------------------------------

fn write_predictInitialization(writer: std.fs.File.Writer) !void {
    const inputs: []TensorZant = try IR_utils.getInputs(tensorZantMap);

    //checks
    if (inputs.len > 1) return error.MoreThanOneInput;
    if (inputs.len < 1) return error.NoInput;

    _ = try writer.print(
        \\  
        \\    //computing the size of the input tensor
        \\    var size: u32 = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        size *= input_shape[dim_i];
        \\    }}
        \\     
        \\    //allocating space in memory for the data
        \\    const data = allocator.alloc(T_in, size) catch return;
        \\    defer allocator.free(data);
        \\    for (0..size) |i| {{
        \\        data[i] = input[i]; // Copying input elements 
        \\    }}
        \\    
        \\    //converting the shape from [*]u32 to []usize
        \\    const usized_shape: []usize = utils.u32ToUsize(allocator, input_shape, shape_len) catch return;
        \\    var tensor_{s} = Tensor(T_in).fromShape(&allocator, @constCast(usized_shape)) catch return;
        \\    defer allocator.free(usized_shape);
        \\    defer tensor_{s}.deinit();
        \\    @memcpy(tensor_{s}.data, data);
    , .{
        try inputs[0].getNameSanitized(),
        try inputs[0].getNameSanitized(),
        try inputs[0].getNameSanitized(),
    });
}

fn writeReturn(writer: std.fs.File.Writer) !void {
    const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

    //checks
    if (outputs.len > 1) return error.MoreThanOneOutput;
    if (outputs.len < 1) return error.NoOutput;

    _ = try writer.print(
        \\
        \\    result.* = tensor_{s}.data.ptr;
        \\
    , .{try outputs[0].getNameSanitized()});

    if (codegen_options.IR_log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Prediction completed.\n")));
            \\    }}
        , .{});
    }
}

// -------------------------------- OTHER WRITE --------------------------------
fn write_TensorShape(writer: std.fs.File.Writer, tz: *TensorZant) !i64 {
    var size: i64 = 1;
    const tensor_shape = tz.getShape();

    try writer.print(
        \\
        \\
        \\var shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{
        try tz.getNameSanitized(),
        tensor_shape.len, // Use adjusted length
    });

    for (tensor_shape, 0..) |dim_i, i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print(
            \\ {}
        , .{dim_i});

        size *= @intCast(dim_i);
    }

    try writer.print(
        \\}} ;
    , .{});

    return size;
}

fn write_checks(writer: std.fs.File.Writer) !void {
    // Autogen a check for the input shape as arg VS input shape as codegen option

    const inputs: []TensorZant = try IR_utils.getInputs(tensorZantMap);

    //checks
    if (inputs.len > 1) return error.MoreThanOneInput;
    if (inputs.len < 1) return error.NoInput;

    //check on the number of dims
    _ = try writer.print(
        \\
        \\    //checks on the input parameters
        \\    if (shape_len == 0) return ;
        \\    if(shape_len != {}) return ;
    , .{inputs[0].getShape().len});

    //check on dims correspondance
    for (inputs[0].getShape(), 0..) |dim, i| {
        _ = try writer.print(
            \\
            \\    if( input_shape[{}] != {}) return ;
        , .{ i, dim });
    }
}

fn write_graphSerialization(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    for (linearizedGraph.items) |node| {
        if (codegen_options.IR_comm) {
            try write_op_info(writer, node);
        }
        if (codegen_options.IR_log) {
            try writer.print(
                \\ 
                \\
                \\    if (log_function) |log| {{
                \\        log(@constCast(@ptrCast("Running {s} operation...\n")));
                \\    }}
            , .{node.*.nodeProto.*.op_type});
        }

        try node.write_op(writer);
    }
}

fn write_op_info(writer: std.fs.File.Writer, node: *NodeZant) !void {
    try writer.print(
        \\
        \\
        \\   //forwarding operation : {s}
    , .{node.*.op_type});

    try writer.print(
        \\
        \\   //parameters:
        \\   //   inputs: 
    , .{});

    // //write the inputs
    // for (node.inputs.items) |input| {
    //     try writer.print(
    //         \\
    //         \\   //      -> {s}
    //     , .{input.name});
    // }
    // try writer.print(
    //     \\
    //     \\   //    outputs:
    // , .{});

    // //write the outputs
    // for (node.outputs.items) |output| {
    //     try writer.print(
    //         \\
    //         \\   //      <- {s}
    //     , .{output.name});
    // }
}
