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

    try write_checks(writer);

    try write_predictInitialization(writer);

    // Allocate output tensors for dynamic mode
    if (codegen_options.dynamic) {
        const output_tensors: []TensorZant = try IR_utils.getOutputs(tensorZantMap);
        for (output_tensors) |*tz| {
            _ = try write_TensorShape(writer, tz);
            const sanitized_name = try tz.getNameSanitized();
            const type_str = tz.ty.toString();
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return;\n", .{ sanitized_name, type_str, sanitized_name });
            //since we are using dynamic inference  we also have to free the output_tensor so to avoid leaks, seee how I return the output tensor in writeReturn()
            try writer.print("    defer tensor_{s}.deinit();", .{sanitized_name});
        }
    }

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
        try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return;", .{ sanitized_name, type_str, sanitized_name });
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

    if (codegen_options.dynamic) {
        _ = try writer.print(
            \\     
            \\     const output_zant_slice = allocator.alloc(T_out, tensor_{s}.size) catch return;
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
    // Add deallocation for dynamic tensors
    if (codegen_options.dynamic) {
        const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);
        for (linkers) |*tz| {
            _ = try writer.print(
                \\    tensor_{s}.deinit();
                \\
            , .{try tz.getNameSanitized()});
        }
    }

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

    //if not used anymore in the rest of the graph
    for (try node.get_output_tensors()) |output_tensor| {
        if (output_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
            _ = try write_TensorShape(
                writer,
                output_tensor,
            );

            const sanitized_name = try output_tensor.getNameSanitized();

            // --- ADD CHECK FOR UNDEFINED TYPE ---
            if (output_tensor.ty == .undefined) {
                std.log.warn("\n\nCODEGEN ERROR: Attempted to generate output tensor '{s}' but its data type is UNDEFINED. Check ONNX graph.\n\n", .{sanitized_name});
                return error.DataTypeNotAvailable; // Or a more specific error like CannotGenerateUndefinedType
            }
            // --- END CHECK ---

            const type_str = output_tensor.ty.toString();

            // Dynamic allocation: Use fromShape to allow mutation
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return;", .{ sanitized_name, type_str, sanitized_name });
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
