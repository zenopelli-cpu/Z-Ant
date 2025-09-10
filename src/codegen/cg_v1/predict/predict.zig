const std = @import("std");
const zant = @import("zant");

const IR_zant = @import("IR_zant");

const IR_codegen = IR_zant.IR_codegen;

// --- zant IR
const IR_utils = IR_zant.utils;
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const NodeZant = IR_zant.NodeZant;

const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_zant.tensorZant_lib.tensorMap;

const allocator = std.heap.page_allocator;

const codegen_options = @import("codegen_options");

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant), do_export: bool) !void {

    // Pre-pass: infer LINK output shapes from operators to replace placeholder [1] shapes
    try infer_link_output_shapes(linearizedGraph);
    // Pre-pass: for DequantizeLinear nodes, align input x LINK shape to y shape if placeholder
    try align_dequant_input_shapes(linearizedGraph);

    // Static initialization for output tensors if not using dynamic allocation
    //
    // declare all the outputs for each node, aka: linkers
    if (!codegen_options.dynamic) try write_linkersInitialization(writer);

    // declare all the outputs of  the network
    try write_outputsInitialization(writer);

    // method to reset the tensors values
    if (!codegen_options.dynamic) try write_linkersResetMethod(writer);

    const inputs = try IR_utils.getInputs(tensorZantMap);
    const outputs = try IR_utils.getOutputs(tensorZantMap);

    //write input type
    if (inputs.len == 0) {
        _ = try writer.print(
            \\
            \\ const T_in : type = {s};
        , .{if (outputs.len > 0) outputs[0].ty.toString() else @as([]const u8, "f32")});
    } else {
        _ = try writer.print(
            \\
            \\ const T_in : type = {s};
        , .{inputs[0].ty.toString()});
    }

    //write output type
    _ = try writer.print(
        \\
        \\ const T_out : type = {s};
    , .{if (outputs.len > 0) outputs[0].ty.toString() else @as([]const u8, "f32")});

    _ = try writer.print(
        \\
        \\ // return codes:
        \\ //  0 : everything good
        \\ // -1 : something when wrong in the mathematical operations
        \\ // -2 : something when wrong in the initialization phase
        \\ // -3 : something when wrong in the output/return phase
        \\pub {s} fn predict (
        \\    input: [*]T_in,
        \\    input_shape: [*]u32,
        \\    shape_len: u32,
        \\    result: *[*]T_out,
        \\) {s} i32 {{
    , .{
        if (do_export == true) "export" else "",
        if (do_export == true) "callconv(.C)" else "",
    });

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Starting prediction...\n")));
            \\    }}
        , .{});
    }

    //if I'm using statical allocation I'll reset all the Link tensors to zero
    if (!codegen_options.dynamic) {
        _ = try writer.print(
            \\
            \\    // Reset all linker tensors to zero before each prediction
            \\    resetOutputTensors();
        , .{});
    }

    // Suppress unused parameter warnings for nodes with no inputs
    if (inputs.len == 0) {
        _ = try writer.print(
            \\
            \\    // Suppress unused parameter warnings for no-input nodes
            \\    _ = input;
            \\    _ = input_shape;
            \\    _ = shape_len;
        , .{});
    }

    try write_checks(writer);

    try write_predictInitialization(writer);

    // Allocate output tensors for dynamic mode
    if (codegen_options.dynamic) {
        const output_tensors: []TensorZant = try IR_utils.getOutputs(tensorZantMap);
        for (output_tensors) |*tz| {
            _ = try write_TensorShape(writer, tz);
            const sanitized_name = try tz.getNameSanitized();
            const type_str = tz.ty.toString();
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return -2;\n", .{ sanitized_name, type_str, sanitized_name });
            //since we are using dynamic inference  we also have to free the output_tensor so to avoid leaks, seee how I return the output tensor in writeReturn()
            try writer.print("    defer tensor_{s}.deinit();", .{sanitized_name});
        }
    }

    try write_graphSerialization(writer, linearizedGraph);

    try writeReturn(writer);

    _ = try writer.print(
        \\
        \\    return 0;
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

fn write_TensorAllocation(writer: std.fs.File.Writer, tz: *TensorZant, size: i64) !void {
    const sanitized_name = try tz.getNameSanitized();

    // --- ADD CHECK FOR UNDEFINED TYPE ---
    if (tz.ty == .undefined) {
        std.log.warn("\n\nCODEGEN ERROR: Attempted to generate output tensor '{s}' but its data type is UNDEFINED. Check ONNX graph analysis in globals.zig.\n\n", .{sanitized_name});
        return error.DataTypeNotAvailable; // Or a more specific error like CannotGenerateUndefinedType
    }
    // --- END CHECK ---

    const type_str = tz.ty.toString();

    if (codegen_options.dynamic) {
        // Dynamic allocation: Use fromShape
        try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return -2;", .{ sanitized_name, type_str, sanitized_name });
    } else {
        // Static allocation: Use fromConstBuffer to allow mutation
        try writer.print("    var array_{s}: [{d}]{s} = [_]{s}{{0}} ** {d};", .{ sanitized_name, size, type_str, type_str, size });
        try writer.print("    var tensor_{s} = Tensor({s}).fromConstBuffer(&fba, &array_{s}, &shape_tensor_{s});", .{ sanitized_name, type_str, sanitized_name, sanitized_name });
    }
}

fn write_linkersResetMethod(writer: std.fs.File.Writer) !void {
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

    // --------- linkers
    const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);

    for (linkers) |*tz| {
        if (!codegen_options.dynamic) {
            _ = try writer.print(
                \\
                \\    @memset(array_{s}[0..], 0);
            , .{try tz.getNameSanitized()});
        }

        if (codegen_options.log) {
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
        if (!codegen_options.dynamic) {
            _ = try writer.print(
                \\
                \\    @memset(array_{s}[0..], 0);
            , .{try tz.getNameSanitized()});
        }

        if (codegen_options.log) {
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
    if (!codegen_options.dynamic) {
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
}

// -------------------------------- WRITE PREDICT() --------------------------------

fn write_predictInitialization(writer: std.fs.File.Writer) !void {
    const inputs: []TensorZant = try IR_utils.getInputs(tensorZantMap);

    // if there are no external inputs (e.g., node extracted model with only initializers), skip input setup
    if (inputs.len == 0) {
        return;
    }

    // Identify the primary external input (first non-initializer)
    var primary_index: usize = std.math.maxInt(usize);
    for (inputs, 0..) |*tz, idx| {
        if (tz.tc != tensorZant_lib.TensorCategory.INITIALIZER) {
            primary_index = idx;
            break;
        }
    }

    if (primary_index == std.math.maxInt(usize)) {
        // No runtime-provided inputs needed
        return;
    }

    //checks
    // Allow multiple inputs; only the primary is sourced from the user pointer
    if (inputs.len > 1) {
        // no-op: other inputs will be allocated below
    }

    _ = try writer.print(
        \\  
        \\    //computing the size of the input tensor
        \\    var size: u32 = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        size *= input_shape[dim_i];
        \\    }}
        \\     
        \\    //allocating space in memory for the data
        \\    const data = allocator.alloc(T_in, size) catch return -2;
        \\    defer allocator.free(data);
        \\    for (0..size) |i| {{
        \\        data[i] = input[i]; // Copying input elements 
        \\    }}
        \\    
        \\    //converting the shape from [*]u32 to []usize
        \\    const usized_shape: []usize = utils.u32ToUsize(allocator, input_shape, shape_len) catch return -2;
        \\    var tensor_{s} = Tensor(T_in).fromShape(&allocator, @constCast(usized_shape)) catch return -2;
        \\    defer allocator.free(usized_shape);
        \\    defer tensor_{s}.deinit();
        \\    @memcpy(tensor_{s}.data, data);
    , .{
        try inputs[primary_index].getNameSanitized(),
        try inputs[primary_index].getNameSanitized(),
        try inputs[primary_index].getNameSanitized(),
    });

    // For any additional non-initializer inputs, allocate zero-initialized tensors using their declared shapes
    for (inputs, 0..) |*tz, idx| {
        if (idx == primary_index) continue;
        if (tz.tc == tensorZant_lib.TensorCategory.INITIALIZER) continue;
        _ = try write_TensorShape(writer, tz);
        const sanitized_name = try tz.getNameSanitized();
        const type_str = tz.ty.toString();
        try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return -2;\n", .{ sanitized_name, type_str, sanitized_name });
        try writer.print("    defer tensor_{s}.deinit();\n", .{sanitized_name});
        try writer.print("    @memset(tensor_{s}.data[0..], 0);\n", .{sanitized_name});
    }
}

fn writeReturn(writer: std.fs.File.Writer) !void {
    const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

    //checks
    if (outputs.len > 1) {
        // For one-op generator, just return the first output and warn
        std.log.warn("Model has {d} outputs; returning only the first ({s}) in generated predict().", .{ outputs.len, try outputs[0].getNameSanitized() });
    }
    if (outputs.len < 1) return error.NoOutput;

    if (codegen_options.dynamic) {
        _ = try writer.print(
            \\     
            \\     const output_zant_slice = allocator.alloc(T_out, tensor_{s}.size) catch return -3;
            \\     @memcpy(output_zant_slice, tensor_{s}.data[0..tensor_{s}.size]);
            \\      
            \\     //The Caller must handle the memory of output_zant_slice
            \\     result.* = output_zant_slice.ptr;
            \\
        , .{ try outputs[0].getNameSanitized(), try outputs[0].getNameSanitized(), try outputs[0].getNameSanitized() });
    } else {
        _ = try writer.print(
            \\
            \\    result.* = tensor_{s}.data.ptr;
            \\
        , .{try outputs[0].getNameSanitized()});
    }

    // Add deallocation for dynamic tensors -> OSS!! not necessary, the tensor are deallocated in deallocate_useless_link_tensors()
    //
    // if (codegen_options.dynamic) {
    //     const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);
    //     for (linkers) |*tz| {
    //         _ = try writer.print(
    //             \\    tensor_{s}.deinit();
    //             \\
    //         , .{try tz.getNameSanitized()});
    //     }
    // }

    if (codegen_options.log) {
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

    // if there are no external inputs, there is nothing to check
    if (inputs.len == 0) {
        return;
    }

    // Allow multiple inputs; only validate against the first non-initializer input
    var check_index: usize = 0;
    for (inputs, 0..) |*tz, idx| {
        if (tz.tc != tensorZant_lib.TensorCategory.INITIALIZER) {
            check_index = idx;
            break;
        }
    }

    //check on the number of dims
    _ = try writer.print(
        \\ 
        \\    //checks on the input parameters
        \\    if (shape_len == 0) return -2;
        \\    if(shape_len != {}) return -2;
    , .{inputs[check_index].getShape().len});

    //check on dims correspondance
    for (inputs[check_index].getShape(), 0..) |dim, i| {
        _ = try writer.print(
            \\
            \\    if( input_shape[{}] != {}) return -2;
        , .{ i, dim });
    }
}

fn write_graphSerialization(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    for (linearizedGraph.items, 0..) |node, i| {
        if (codegen_options.comm) {
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

        //Before computing the OP, init link tensors when we are in dynamic allocation
        if (codegen_options.dynamic) try allocate_output_link_tensors(writer, node); //ERE YOU ALLOCATE

        try node.write_op(writer);

        //After computing the OP, delete link tensors that are not useful anymore when we are in dynamic allocation
        if (codegen_options.dynamic) try deallocate_useless_link_tensors(writer, i, linearizedGraph); //HERE YOU DEALLOCATE
    }
}

//dynamically allocate a linker tensor
fn allocate_output_link_tensors(writer: std.fs.File.Writer, node: *NodeZant) !void {

    // Attempt to infer output shape(s) from the operator before allocating LINK tensors
    var inferred_shape: ?[]usize = null;
    inferred_shape = node.op.get_output_shape() catch null;

    if (inferred_shape) |_| {} else {}

    //if not used anymore in the rest of the graph
    for (try node.get_output_tensors()) |output_tensor| {
        if (output_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
            // If the current shape is a placeholder like [1], and we inferred a better shape, apply it
            if (inferred_shape) |shape_out| {
                if (output_tensor.shape.len == 1 and output_tensor.shape[0] == 1) {
                    output_tensor.shape = shape_out;
                    output_tensor.stride = try tensorZant_lib.TensorZant.computeStride(shape_out);
                } else {}
            } else {}

            const tensor_size = try write_TensorShape(
                writer,
                output_tensor,
            );
            _ = tensor_size;

            const sanitized_name = try output_tensor.getNameSanitized();

            // --- ADD CHECK FOR UNDEFINED TYPE ---
            if (output_tensor.ty == .undefined) {
                std.log.warn("\n\nCODEGEN ERROR: Attempted to generate output tensor '{s}' but its data type is UNDEFINED. Check ONNX graph.\n\n", .{sanitized_name});
                return error.DataTypeNotAvailable; // Or a more specific error like CannotGenerateUndefinedType
            }
            // --- END CHECK ---

            const type_str = output_tensor.ty.toString();

            // Dynamic allocation: Use fromShape to allow mutation
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return -2;", .{ sanitized_name, type_str, sanitized_name });
        }
    }
}

//free a linker tensor after dynamical allocation
fn deallocate_useless_link_tensors(writer: std.fs.File.Writer, starting_node: usize, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    const node = linearizedGraph.items[starting_node];

    //After computing the OP, delete link tensors that are not useful anymore when we are in dynamic allocation

    var used: bool = false;
    for (try node.get_input_tensors()) |my_input_tensor| { //for each input tensor of my node
        used = false;
        //if not used anymore in the rest of the graph
        for (starting_node + 1..linearizedGraph.items.len) |j| {
            for (try linearizedGraph.items[j].get_input_tensors()) |other_input_tens| {
                if (std.mem.eql(u8, my_input_tensor.name, other_input_tens.name)) {
                    used = true;
                    break;
                }
            }
            if (used) break;
        }

        //if it is not used anymore in the graph and it is an initializer I can deinit it
        if (!used and my_input_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
            _ = try writer.print("    tensor_{s}.deinit();\n", .{try my_input_tensor.getNameSanitized()});
        }
    }

    //
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

    //write the inputs
    for (try node.get_input_tensors()) |input| {
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
    for (try node.get_output_tensors()) |output| {
        try writer.print(
            \\
            \\   //      <- {s}
        , .{output.name});
    }
}

// Helper to compare shapes
fn shapeEqual(a: []const usize, b: []const usize) bool {
    if (a.len != b.len) return false;
    for (a, b) |a_val, b_val| {
        if (a_val != b_val) return false;
    }
    return true;
}

// Perform a pre-pass over nodes to infer and assign LINK tensors' shapes
fn infer_link_output_shapes(linearizedGraph: std.ArrayList(*NodeZant)) !void {
    for (linearizedGraph.items) |node| {
        // Try to compute the output shape for this node
        const inferred_shape = node.op.get_output_shape() catch null;
        if (inferred_shape == null) continue;

        // FORCE update ALL LINK outputs - no conditions
        for (try node.get_output_tensors()) |tz| {
            if (tz.tc == tensorZant_lib.TensorCategory.LINK) {
                _ = tz.shape;
                tz.shape = inferred_shape.?;
                tz.stride = try tensorZant_lib.TensorZant.computeStride(tz.shape);
            }
        }
    }
}

// For DequantizeLinear nodes: if input x is a LINK with placeholder shape, set it to y's shape
fn align_dequant_input_shapes(linearizedGraph: std.ArrayList(*NodeZant)) !void {
    for (linearizedGraph.items) |node| {
        if (std.mem.eql(u8, node.op_type, "DequantizeLinear")) {
            const inputs = try node.get_input_tensors();
            const outputs = try node.get_output_tensors();
            if (outputs.len == 0) continue;
            const y = outputs[0];
            if (inputs.len >= 1) {
                const x = inputs[0];
                if (x.tc == tensorZant_lib.TensorCategory.LINK) {
                    const x_len = x.shape.len;
                    const y_len = y.shape.len;
                    var mismatch = (x_len != y_len);
                    if (!mismatch) {
                        // check dims differ
                        for (x.shape, 0..) |xd, i| {
                            if (xd != y.shape[i]) {
                                mismatch = true;
                                break;
                            }
                        }
                    }
                    if (mismatch) {
                        x.shape = y.shape;
                        x.stride = try tensorZant_lib.TensorZant.computeStride(x.shape);
                    }
                }
            }
        }
        // Force fix for QCONV that feeds into QLinearSoftmax: override hardcoded output shape
        if (std.mem.eql(u8, node.op_type, "QLinearConv")) {
            const outputs = try node.get_output_tensors();
            if (outputs.len >= 1) {
                const output_tensor = outputs[0];
                if (output_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
                    if (std.mem.containsAtLeast(u8, output_tensor.name, 1, "class_map")) {
                        // This is the final conv that feeds softmax - force correct shape inference
                        _ = output_tensor.shape;
                        const corrected_shape = try node.op.get_output_shape();
                        output_tensor.shape = corrected_shape;
                        output_tensor.stride = try tensorZant_lib.TensorZant.computeStride(output_tensor.shape);
                    }
                }
            }
        }

        // Force fix for QLinearSoftmax input: find final OUTPUT and propagate its shape back to softmax input
        if (std.mem.eql(u8, node.op_type, "QLinearSoftmax")) {
            const inputs = try node.get_input_tensors();
            if (inputs.len >= 1) {
                const input_tensor = inputs[0];
                if (input_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
                    // Find the final OUTPUT tensor shape from the graph and use it for softmax input
                    for (linearizedGraph.items) |other_node| {
                        const other_outputs = try other_node.get_output_tensors();
                        for (other_outputs) |ot| {
                            if (ot.tc == tensorZant_lib.TensorCategory.OUTPUT and ot.shape.len >= 2) {
                                // Found the final output - use its shape for softmax input but add batch dim if missing
                                if (ot.shape.len == 3) {
                                    // Convert {2,12,12} to {1,2,12,12} to match conv expectations
                                    const new_shape = [_]usize{ 1, ot.shape[0], ot.shape[1], ot.shape[2] };
                                    const target_shape = try allocator.dupe(usize, new_shape[0..]);
                                    input_tensor.shape = target_shape;
                                } else {
                                    // Use original shape, duplicate to remove const
                                    const target_shape = try allocator.dupe(usize, ot.shape);
                                    input_tensor.shape = target_shape;
                                }
                                input_tensor.stride = try tensorZant_lib.TensorZant.computeStride(input_tensor.shape);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}
