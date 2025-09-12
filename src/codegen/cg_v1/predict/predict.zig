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

// For pattern detection
const Clip = IR_zant.operators.Clip;

const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_zant.tensorZant_lib.tensorMap;

const allocator = std.heap.page_allocator;

const codegen_options = @import("codegen_options");

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant), do_export: bool) !void {

    // Pre-pass: infer LINK output shapes from operators to replace placeholder [1] shapes
    try infer_link_output_shapes(linearizedGraph);
    // Pre-pass: for DequantizeLinear nodes, align input x LINK shape to y shape if placeholder
    try align_dequant_input_shapes(linearizedGraph);

    // REMOVED: Static initialization for output tensors - now using just-in-time allocation
    // declare all the outputs for each node, aka: linkers
    // if (!codegen_options.dynamic) try write_linkersInitialization(writer);

    // declare all the outputs of  the network
    try write_outputsInitialization(writer);

    // REMOVED: method to reset the tensors values - no longer needed with JIT allocation
    // if (!codegen_options.dynamic) try write_linkersResetMethod(writer);

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

    // Allocate output tensors for dynamic/static mode
    const output_tensors: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

    // Declare output_active_is_a once for static mode
    if (!codegen_options.dynamic and output_tensors.len > 0) {
        try writer.print("    const output_active_is_a = fba_live_a <= fba_live_b;\n", .{});
    }

    for (output_tensors) |*tz| {
        const sanitized_name = try tz.getNameSanitized();
        const type_str = tz.ty.toString();

        if (codegen_options.dynamic) {
            _ = try write_TensorShape(writer, tz);
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return -2;\n", .{ sanitized_name, type_str, sanitized_name });
            //since we are using dynamic inference  we also have to free the output_tensor so to avoid leaks, seee how I return the output tensor in writeReturn()
            try writer.print("    defer tensor_{s}.deinit();", .{sanitized_name});
        } else {
            // Static mode: Create tensor from pre-allocated array using dual FBA pools
            try writer.print("    var tensor_{s} = Tensor({s}).fromConstBuffer(\n", .{ sanitized_name, type_str });
            try writer.print("        if (output_active_is_a) &fba_a else &fba_b,\n", .{});
            try writer.print("        &array_{s},\n", .{sanitized_name});
            try writer.print("        &shape_tensor_{s}\n", .{sanitized_name});
            try writer.print("    );\n", .{});
        }
    }

    // Build alias map and reference counts for smart deallocation (both dynamic and static)
    if (codegen_options.dynamic or !codegen_options.dynamic) {
        var alias_map = try buildAliasMap(allocator, linearizedGraph);
        defer {
            var alias_iter = alias_map.iterator();
            while (alias_iter.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            alias_map.deinit();
        }

        var use_counts = try buildUseCounts(allocator, linearizedGraph, &alias_map);
        defer {
            var count_iter = use_counts.iterator();
            while (count_iter.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            use_counts.deinit();
        }

        // set dei contatori già emessi (solo chi decrementiamo)
        var rc_declared = std.StringHashMap(u8).init(allocator);
        defer {
            var it = rc_declared.iterator();
            while (it.next()) |e| allocator.free(e.key_ptr.*);
            rc_declared.deinit();
        }

        try write_graphSerialization(writer, linearizedGraph, &alias_map, &use_counts, &rc_declared, allocator);
    } else {
        try write_graphSerialization(writer, linearizedGraph, null, null, null, allocator);
    }

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
        try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return -2;\n", .{ sanitized_name, type_str, sanitized_name });
        // Add defer for intermediate tensors (not for final outputs which are handled by reference counting)
        try writer.print("    defer tensor_{s}.deinit();\n", .{sanitized_name});
    } else {
        // Static allocation: Use fromConstBuffer to allow mutation with ping-pong pools
        // Static allocation: zero-init esplicito con tipo
        try writer.print("var array_{s}: [{d}]{s} = std.mem.zeroes([{d}]{s});\n", .{ sanitized_name, size, type_str, size, type_str });

        // Scope per costruire il Tensor sull'array pre-allocato

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

    // Calculate input size and generate zero-copy tensor initialization
    const input_shape = inputs[primary_index].getShape();
    var input_size: u64 = 1;
    for (input_shape) |dim| {
        input_size *= dim;
    }

    _ = try writer.print(
        \\    
        \\    // Fixed input shape (zero-copy optimization)
        \\    var input_shape_fixed: [{}]usize = .{{ 
    , .{input_shape.len});

    for (input_shape, 0..) |dim, i| {
        if (i > 0) try writer.print(", ", .{});
        try writer.print("{}", .{dim});
    }

    _ = try writer.print(
        \\ }};
        \\    const input_size: usize = {};
        \\    
        \\    // Zero-copy tensor pointing directly to input data
        \\    const tensor_{s} = Tensor(T_in){{
        \\        .data = input[0..input_size],
        \\        .shape = input_shape_fixed[0..],
        \\        .size = input_size,
        \\        .allocator = &allocator, // doesn't manage memory
        \\    }};
        \\
    , .{
        input_size,
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
            \\     // Track allocation size for safe deallocation
            \\     last_result_size = tensor_{s}.size;
            \\      
            \\     //The Caller must handle the memory of output_zant_slice
            \\     result.* = output_zant_slice.ptr;
            \\
        , .{ try outputs[0].getNameSanitized(), try outputs[0].getNameSanitized(), try outputs[0].getNameSanitized(), try outputs[0].getNameSanitized() });
    } else {
        _ = try writer.print(
            \\
            \\    result.* = tensor_{s}.data.ptr;
            \\    last_result_size = 0; // Static allocation, no deallocation needed
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
        \\
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

fn write_graphSerialization(
    writer: std.fs.File.Writer,
    linearizedGraph: std.ArrayList(*NodeZant),
    alias_map: ?*const std.StringHashMap([]const u8),
    use_counts: ?*const std.StringHashMap(u32),
    rc_declared: ?*std.StringHashMap(u8),
    alloc: std.mem.Allocator,
) !void {
    // Track nodes to skip (e.g., QuantizeLinear consumed by a fused op earlier)
    var skip_nodes = std.AutoHashMap(*NodeZant, void).init(alloc);
    defer skip_nodes.deinit();

    // Pre-plan QAdd fusions: map Add nodes to fusion pattern
    var qadd_plan = std.AutoHashMap(*NodeZant, QAddPatternResult).init(alloc);
    defer qadd_plan.deinit();
    {
        var k: usize = 0;
        while (k < linearizedGraph.items.len) : (k += 1) {
            const nd = linearizedGraph.items[k];
            if (!std.mem.eql(u8, nd.op_type, "Add")) continue;
            const add_in = nd.get_input_tensors() catch &[_]*TensorZant{};
            const add_out = nd.get_output_tensors() catch &[_]*TensorZant{};
            if (add_in.len < 2 or add_out.len < 1) continue;
            // Backward: find dequant producers for both inputs
            const deq_a = resolve_dequant_producer(linearizedGraph, k, add_in[0].name);
            const deq_b = resolve_dequant_producer(linearizedGraph, k, add_in[1].name);
            if (deq_a == null or deq_b == null) continue;
            // Forward: find unique QuantizeLinear consumer for Add output (through pass-throughs)
            const qfind = find_quantize_consumer(linearizedGraph, k, add_out[0].name);
            if (qfind == null) continue;
            // Safety: ensure no other future consumers of the Add output after the found quantize
            var unique_use = true;
            var jcheck: usize = qfind.?.index + 1;
            while (jcheck < linearizedGraph.items.len) : (jcheck += 1) {
                const fut = linearizedGraph.items[jcheck];
                const fut_inputs = fut.get_input_tensors() catch &[_]*TensorZant{};
                for (fut_inputs) |tin| {
                    if (std.mem.eql(u8, tin.name, add_out[0].name)) {
                        unique_use = false;
                        break;
                    }
                }
                if (!unique_use) break;
            }
            if (!unique_use) continue;
            // Build pattern
            const deq_a_in = deq_a.?.get_input_tensors() catch &[_]*TensorZant{};
            const deq_b_in = deq_b.?.get_input_tensors() catch &[_]*TensorZant{};
            if (deq_a_in.len < 3 or deq_b_in.len < 3) continue;
            const q_in = qfind.?.node.get_input_tensors() catch &[_]*TensorZant{};
            const q_out = qfind.?.node.get_output_tensors() catch &[_]*TensorZant{};
            if (q_in.len < 3 or q_out.len < 1) continue;
            var pat = QAddPatternResult{ .detected = true };
            pat.deq_a_node = deq_a.?;
            pat.deq_b_node = deq_b.?;
            pat.add_node = nd;
            pat.quant_node = qfind.?.node;
            pat.a_q = deq_a_in[0];
            pat.a_scale = deq_a_in[1];
            pat.a_zp = deq_a_in[2];
            pat.b_q = deq_b_in[0];
            pat.b_scale = deq_b_in[1];
            pat.b_zp = deq_b_in[2];
            pat.c_q = q_out[0];
            pat.c_scale = q_in[1];
            pat.c_zp = q_in[2];
            // Record and mark nodes to skip later
            _ = try qadd_plan.put(nd, pat);
            _ = try skip_nodes.put(deq_a.?, {});
            _ = try skip_nodes.put(deq_b.?, {});
            _ = try skip_nodes.put(qfind.?.node, {});
        }
    }

    var i: usize = 0;
    while (i < linearizedGraph.items.len) {
        const node = linearizedGraph.items[i];

        // Skip nodes marked as handled by previous fusions
        if (skip_nodes.contains(node)) {
            i += 1;
            continue;
        }

        // Pattern detection: Fused NHWC f32 -> NCHW u8 (Transpose -> QuantizeLinear) at graph head
        if (i + 1 < linearizedGraph.items.len) {
            const tq_pat = try detect_transpose_quantize_pattern(linearizedGraph.items[i .. i + 2]);
            if (tq_pat.detected) {
                // Allocate output buffer for quantized tensor (node2 output)
                if (codegen_options.dynamic) {
                    try allocate_output_link_tensors(writer, tq_pat.quant_node.?);
                } else {
                    try allocate_output_link_tensors_static(writer, tq_pat.quant_node.?, linearizedGraph, i + 1, alias_map, use_counts);
                }

                // Emit fused quantizing transpose
                try write_transpose_quantize_fused(writer, tq_pat);

                // Skip the 2 operations we just optimized
                i += 2;
                continue;
            }
        }

        // Pattern detection: DequantizeLinear -> DequantizeLinear -> Add -> QuantizeLinear (QAdd fusion)
        if (i + 3 < linearizedGraph.items.len) {
            const qadd_pat = try detect_qadd_pattern(linearizedGraph.items[i .. i + 4]);
            if (qadd_pat.detected) {
                // Safety: ensure Add output is only consumed by this QuantizeLinear
                var add_out_safe = true;
                const add_out_tensors = try qadd_pat.add_node.?.get_output_tensors();
                if (add_out_tensors.len > 0) {
                    const add_out_name = add_out_tensors[0].name;
                    // scan future nodes beyond the quantize (i + 3)
                    var j: usize = i + 4;
                    while (j < linearizedGraph.items.len) : (j += 1) {
                        const fut = linearizedGraph.items[j];
                        const fut_inputs = fut.get_input_tensors() catch &[_]*TensorZant{};
                        for (fut_inputs) |tin| {
                            if (std.mem.eql(u8, tin.name, add_out_name)) {
                                add_out_safe = false;
                                break;
                            }
                        }
                        if (!add_out_safe) break;
                    }
                }

                if (add_out_safe) {
                    // Allocate only the final quantized output buffer
                    if (codegen_options.dynamic) {
                        try allocate_output_link_tensors(writer, qadd_pat.quant_node.?);
                    } else {
                        try allocate_output_link_tensors_static(writer, qadd_pat.quant_node.?, linearizedGraph, i + 3, alias_map, use_counts);
                    }
                    try write_qadd_fused(writer, qadd_pat);
                    i += 4;
                    continue;
                }
            }
        }

        // Add-anchored QAdd fusion: Add -> QuantizeLinear, with inputs from two DequantizeLinear producers anywhere before
        if (std.mem.eql(u8, node.op_type, "Add")) {
            const add_node = linearizedGraph.items[i];
            const add_in = try add_node.get_input_tensors();
            const add_out = try add_node.get_output_tensors();
            if (add_in.len >= 2 and add_out.len >= 1) {
                // Find the QuantizeLinear that consumes this Add output, traversing pass-through ops
                var quant_idx: ?usize = null;
                var quant_node_ptr: ?*NodeZant = null;
                var jfind: usize = i + 1;
                var current_name = add_out[0].name;
                outer: while (jfind < linearizedGraph.items.len) : (jfind += 1) {
                    const cand = linearizedGraph.items[jfind];
                    // If another non-pass-through op consumes current_name, abort fusion
                    const cand_inputs = cand.get_input_tensors() catch &[_]*TensorZant{};
                    var consumes_current = false;
                    for (cand_inputs) |cin| {
                        if (std.mem.eql(u8, cin.name, current_name)) {
                            consumes_current = true;
                            break;
                        }
                    }
                    if (consumes_current) {
                        if (std.mem.eql(u8, cand.op_type, "QuantizeLinear")) {
                            const q_in = try cand.get_input_tensors();
                            if (q_in.len >= 1 and std.mem.eql(u8, q_in[0].name, current_name)) {
                                quant_idx = jfind;
                                quant_node_ptr = cand;
                                break :outer;
                            }
                        } else if (std.mem.eql(u8, cand.op_type, "Identity") or std.mem.eql(u8, cand.op_type, "Reshape") or std.mem.eql(u8, cand.op_type, "Squeeze") or std.mem.eql(u8, cand.op_type, "Unsqueeze") or std.mem.eql(u8, cand.op_type, "Transpose")) {
                            const cand_outs = cand.get_output_tensors() catch &[_]*TensorZant{};
                            if (cand_outs.len == 0) break :outer;
                            current_name = cand_outs[0].name;
                            continue;
                        } else {
                            // A different op consumes the Add output; can't fuse safely
                            break :outer;
                        }
                    }
                }

                if (quant_idx) |q_idx| {
                    const quant_node = quant_node_ptr.?;
                    const q_in = try quant_node.get_input_tensors();
                    const q_out = try quant_node.get_output_tensors();

                    // Safety: ensure no other future consumer besides this Quantize
                    var add_out_unique = true;
                    var jcheck: usize = q_idx + 1;
                    while (jcheck < linearizedGraph.items.len) : (jcheck += 1) {
                        const fut = linearizedGraph.items[jcheck];
                        const fut_inputs = fut.get_input_tensors() catch &[_]*TensorZant{};
                        for (fut_inputs) |tin| {
                            if (std.mem.eql(u8, tin.name, add_out[0].name)) {
                                add_out_unique = false;
                                break;
                            }
                        }
                        if (!add_out_unique) break;
                    }

                    if (add_out_unique and q_in.len >= 3 and q_out.len >= 1) {
                        // scan backwards to find DequantizeLinear producers for add inputs
                        var cand = QAddPatternResult{ .detected = false };
                        var found_a: bool = false;
                        var found_b: bool = false;
                        var j: isize = @intCast(i);
                        while (j >= 0 and !(found_a and found_b)) : (j -= 1) {
                            const prev = linearizedGraph.items[@intCast(j)];
                            if (!std.mem.eql(u8, prev.op_type, "DequantizeLinear")) continue;
                            const prev_out = try prev.get_output_tensors();
                            if (prev_out.len == 0) continue;
                            const out_name = prev_out[0].name;
                            if (!found_a and std.mem.eql(u8, out_name, add_in[0].name)) {
                                const deq_in = try prev.get_input_tensors();
                                if (deq_in.len >= 3) {
                                    cand.deq_a_node = prev;
                                    cand.a_q = deq_in[0];
                                    cand.a_scale = deq_in[1];
                                    cand.a_zp = deq_in[2];
                                    found_a = true;
                                }
                                continue;
                            }
                            if (!found_b and std.mem.eql(u8, out_name, add_in[1].name)) {
                                const deq_in = try prev.get_input_tensors();
                                if (deq_in.len >= 3) {
                                    cand.deq_b_node = prev;
                                    cand.b_q = deq_in[0];
                                    cand.b_scale = deq_in[1];
                                    cand.b_zp = deq_in[2];
                                    found_b = true;
                                }
                                continue;
                            }
                        }

                        if (found_a and found_b) {
                            cand.detected = true;
                            cand.add_node = add_node;
                            cand.quant_node = quant_node;
                            cand.c_q = q_out[0];
                            cand.c_scale = q_in[1];
                            cand.c_zp = q_in[2];

                            if (codegen_options.dynamic) {
                                try allocate_output_link_tensors(writer, cand.quant_node.?);
                            } else {
                                try allocate_output_link_tensors_static(writer, cand.quant_node.?, linearizedGraph, q_idx, alias_map, use_counts);
                            }
                            try write_qadd_fused(writer, cand);
                            _ = try skip_nodes.put(quant_node, {});
                            i += 1;
                            continue;
                        }
                    }
                }
            }
        }

        // Pattern detection: DequantizeLinear -> Clip -> QuantizeLinear
        if (i + 2 < linearizedGraph.items.len) {
            const pattern_result = try detect_quantized_clip_pattern(linearizedGraph.items[i .. i + 3]);
            if (pattern_result.detected) {
                // Note: Output tensor allocation is handled by write_quantized_clip_pattern

                // Write the optimized pattern
                try write_quantized_clip_pattern(writer, pattern_result);

                // NIENTE RC qui: decrementeremo quando il prossimo "vero" consumatore userà il buffer
                i += 3; // Skip the 3 operations we just optimized
                continue;
            }
        }

        // Normal operation writing
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
        if (codegen_options.dynamic) {
            try allocate_output_link_tensors(writer, node); //DYNAMIC ALLOCATION
        } else {
            try allocate_output_link_tensors_static(writer, node, linearizedGraph, i, alias_map, use_counts); //STATIC JIT ALLOCATION
        }

        try node.write_op(writer);
        try writer.print("\n", .{}); // Ensure newline after operation

        // RC decrement "lazy" solo per nodi realmente emessi
        if (codegen_options.dynamic and alias_map != null and use_counts != null and rc_declared != null) {
            try emitRcDecForNode(writer, linearizedGraph.items[i], alias_map.?, use_counts.?, rc_declared.?, alloc); //REFERENCE COUNTING DEALLOCATION
        } else if (codegen_options.dynamic) {
            try deallocate_useless_link_tensors(writer, i, linearizedGraph); //FALLBACK DYNAMIC DEALLOCATION
        } else {
            // STATIC MODE: Apply reference counting mechanism like dynamic but with FBA tracking
            if (alias_map != null and use_counts != null and rc_declared != null) {
                try emitRcDecForNodeStatic(writer, linearizedGraph.items[i], alias_map.?, use_counts.?, rc_declared.?, alloc); //REFERENCE COUNTING FOR STATIC
            } else {
                try reset_fba_after_unused_tensors(writer, i, linearizedGraph); //FALLBACK STATIC FBA RESET
            }
        }

        i += 1;
    }
}

// Resolve the DequantizeLinear producer for a given tensor name by scanning backwards and skipping pass-through ops
fn resolve_dequant_producer(graph: std.ArrayList(*NodeZant), start_index: usize, target_name: []const u8) ?*NodeZant {
    var name = target_name;
    var j: isize = @intCast(start_index);
    while (j >= 0) : (j -= 1) {
        const n = graph.items[@intCast(j)];
        const outs = n.get_output_tensors() catch &[_]*TensorZant{};
        if (outs.len == 0) continue;
        if (!std.mem.eql(u8, outs[0].name, name)) continue;
        if (std.mem.eql(u8, n.op_type, "DequantizeLinear")) return n;
        if (std.mem.eql(u8, n.op_type, "Identity") or
            std.mem.eql(u8, n.op_type, "Reshape") or
            std.mem.eql(u8, n.op_type, "Squeeze") or
            std.mem.eql(u8, n.op_type, "Unsqueeze") or
            std.mem.eql(u8, n.op_type, "Transpose"))
        {
            const ins = n.get_input_tensors() catch &[_]*TensorZant{};
            if (ins.len == 0) return null;
            name = ins[0].name;
            continue;
        }
        return null;
    }
    return null;
}

const QuantConsume = struct { index: usize, node: *NodeZant };
fn find_quantize_consumer(graph: std.ArrayList(*NodeZant), start_index: usize, start_name: []const u8) ?QuantConsume {
    var name = start_name;
    var j: usize = start_index + 1;
    while (j < graph.items.len) : (j += 1) {
        const n = graph.items[j];
        const ins = n.get_input_tensors() catch &[_]*TensorZant{};
        var consumes = false;
        for (ins) |tin| {
            if (std.mem.eql(u8, tin.name, name)) {
                consumes = true;
                break;
            }
        }
        if (!consumes) continue;
        if (std.mem.eql(u8, n.op_type, "QuantizeLinear")) {
            return QuantConsume{ .index = j, .node = n };
        }
        if (std.mem.eql(u8, n.op_type, "Identity") or std.mem.eql(u8, n.op_type, "Reshape") or std.mem.eql(u8, n.op_type, "Squeeze") or std.mem.eql(u8, n.op_type, "Unsqueeze") or std.mem.eql(u8, n.op_type, "Transpose")) {
            const outs = n.get_output_tensors() catch &[_]*TensorZant{};
            if (outs.len == 0) return null;
            name = outs[0].name;
            continue;
        }
        return null;
    }
    return null;
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

            // PRE-DEALLOCATION DISABLED: Too dangerous, causes use-after-free bugs
            // The aggressive deallocation after operations is much safer
            // Example bug: tensor_relu__37_0_quantized.deinit() before it's used in QLinearConv

            // Dynamic allocation: Use fromShape to allow mutation
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return -2;\n", .{ sanitized_name, type_str, sanitized_name });
            // Add defer for intermediate tensors - reference counting will handle proper cleanup
            try writer.print("    defer tensor_{s}.deinit();\n", .{sanitized_name});
        }
    }
}

//statically allocate a linker tensor using FBA (just-in-time)
fn allocate_output_link_tensors_static(
    writer: std.fs.File.Writer,
    node: *NodeZant,
    linearizedGraph: std.ArrayList(*NodeZant),
    current_index: usize,
    alias_map_opt: ?*const std.StringHashMap([]const u8),
    use_counts_opt: ?*const std.StringHashMap(u32),
) !void {
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
                return error.DataTypeNotAvailable;
            }
            // --- END CHECK ---

            const type_str = output_tensor.ty.toString();

            // Decide if RC machinery will ever reference this tensor (to avoid unused constants)
            var will_be_tracked: bool = false;
            if (use_counts_opt) |use_counts_ptr| {
                const canon_name = if (alias_map_opt) |am|
                    canonical(am, sanitized_name)
                else
                    sanitized_name;
                const cnt = use_counts_ptr.*.get(canon_name) orelse 0;
                will_be_tracked = (cnt > 0);
            } else {
                // Fallback: scan only future nodes
                for (current_index + 1..linearizedGraph.items.len) |j| {
                    const future_node = linearizedGraph.items[j];
                    for (try future_node.get_input_tensors()) |fut_in| {
                        if (std.mem.eql(u8, fut_in.name, output_tensor.name)) {
                            will_be_tracked = true;
                            break;
                        }
                    }
                    if (will_be_tracked) break;
                }
            }

            // Static allocation using FBA: Use fromShape for heap allocation in FBA
            try writer.print("    // choose active pool\n", .{});
            try writer.print("    const link_active_is_a_{s} = fba_live_a <= fba_live_b;\n", .{sanitized_name});

            // FIXME: Debug logging removed due to emoji/escaping issues
            // if (codegen_options.log) {
            //     try writer.print("    if (log_function) |log| {{ log(@constCast(@ptrCast(\\\"DEBUG: Choosing FBA pool\\\\n\\\"))); }}\n", .{});
            // }
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(if (link_active_is_a_{s}) &fba_a else &fba_b, &shape_tensor_{s}) catch return -2;\n", .{ sanitized_name, type_str, sanitized_name, sanitized_name });

            // Record live FBA tensor per pool only if RC will track and later decrement it
            if (will_be_tracked) {
                try writer.print("    if (tensor_{s}.allocator == &fba_a) {{ fba_live_a += 1; }} else {{ fba_live_b += 1; }} // +1 live: {s}\n", .{ sanitized_name, sanitized_name });
            }

            // Memory Analytics disabilitato per ora (nomi troppo lunghi)
            _ = codegen_options;
        }
    }
}

//reset FBA when tensors are no longer needed (static allocation optimization)
fn reset_fba_after_unused_tensors(writer: std.fs.File.Writer, starting_node: usize, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    if (starting_node >= linearizedGraph.items.len) return;

    const node = linearizedGraph.items[starting_node];

    var any_dead: bool = false;
    for (try node.get_input_tensors()) |input_tensor| {
        if (input_tensor.tc != tensorZant_lib.TensorCategory.LINK) continue;

        var used_later = false;
        // STRATEGIA 1: Lifetime più aggressivo - controlla solo le prossime 3-5 operazioni invece di tutto il grafo
        const lookahead_limit = @min(starting_node + 5, linearizedGraph.items.len);

        // OTTIMIZZAZIONE FUTURA: Multi-Pool Strategy (3+ pools)
        // Invece di 2 pool da 512KB, usare 4 pool da 128KB:
        // - Pool specializzati per tensor size (small/medium/large)
        // - Pool temporanei per operazioni brevi (clip, relu)
        // - Pool persistenti per tensori long-lived (add, residual connections)
        // Vantaggi: Memory footprint 4×128KB = 512KB vs attuale 2×512KB = 1024KB
        for (starting_node + 1..lookahead_limit) |j| {
            const fut = linearizedGraph.items[j];
            for (try fut.get_input_tensors()) |fut_in| {
                if (std.mem.eql(u8, input_tensor.name, fut_in.name)) {
                    used_later = true;
                    break;
                }
            }
            if (used_later) break;
        }

        if (!used_later) {
            // STRATEGIA 2: Log più dettagliato per debugging
            // FIXME: Temporarily disabled due to long tensor names causing newline issues
            // if (codegen_options.log) {
            //     try writer.print("    if (log_function) |log| {{ log(@constCast(@ptrCast(\"�� Tensor dead: {s}\\n\"))); }}\n", .{input_tensor.name});
            // }
            // We don't know which pool the tensor belongs to; assume it was taken from the pool with greater live count
            try writer.print("    if (fba_live_a >= fba_live_b and fba_live_a > 0) {{ fba_live_a -= 1; }} else if (fba_live_b > 0) {{ fba_live_b -= 1; }}\n", .{});
            any_dead = true;
        }
    }

    if (!any_dead) return;

    // STRATEGIA 3: Reset più aggressivo - anche con soglia bassa invece di aspettare 0
    try writer.print("    if (fba_live_a == 0) {{\n", .{});
    if (codegen_options.log) {
        try writer.print("        if (log_function) |log| {{ log(@constCast(@ptrCast(\"🗑️  FBA Reset(A): pool A empty, reclaiming buffer\\n\"))); }}\n", .{});
    }
    try writer.print("        fba_state_a.reset();\n", .{});
    try writer.print("        fba_live_a = 0;\n", .{});
    try writer.print("    }}\n", .{});

    try writer.print("    if (fba_live_b == 0) {{\n", .{});
    if (codegen_options.log) {
        try writer.print("        if (log_function) |log| {{ log(@constCast(@ptrCast(\"🗑️  FBA Reset(B): pool B empty, reclaiming buffer\\n\"))); }}\n", .{});
    }
    try writer.print("        fba_state_b.reset();\n", .{});
    try writer.print("        fba_live_b = 0;\n", .{});
    try writer.print("    }}\n", .{});
}

// Build alias map: alias_name -> canonical_name for in-place clip operations
fn buildAliasMap(temp_allocator: std.mem.Allocator, graph: std.ArrayList(*NodeZant)) !std.StringHashMap([]const u8) {
    var alias_map = std.StringHashMap([]const u8).init(temp_allocator);

    // Look for optimized quantized clip patterns
    var i: usize = 0;
    while (i + 2 < graph.items.len) : (i += 1) {
        const nodes_slice = graph.items[i .. i + 3];
        const pattern = try detect_quantized_clip_pattern(nodes_slice);
        if (pattern.detected and pattern.output_quantized_tensor != null and pattern.input_quantized_tensor != null) {
            // Map the clip alias to its canonical input tensor
            const alias_name = try pattern.output_quantized_tensor.?.getNameSanitized();
            const canonical_name = try pattern.input_quantized_tensor.?.getNameSanitized();

            const alias_key = try temp_allocator.dupe(u8, alias_name);
            const canonical_value = try temp_allocator.dupe(u8, canonical_name);
            try alias_map.put(alias_key, canonical_value);

            // Skip the pattern nodes since we've processed them
            i += 2;
        }
    }

    return alias_map;
}

// Get canonical tensor name (resolves aliases to their real buffer)
fn canonical(alias_map: *const std.StringHashMap([]const u8), tensor_name: []const u8) []const u8 {
    if (alias_map.get(tensor_name)) |canonical_name| {
        return canonical_name;
    }
    return tensor_name;
}

// Build reference count map for canonical tensors only, skipping optimized patterns
fn buildUseCounts(
    temp_allocator: std.mem.Allocator,
    graph: std.ArrayList(*NodeZant),
    alias_map: *const std.StringHashMap([]const u8),
) !std.StringHashMap(u32) {
    var counts = std.StringHashMap(u32).init(temp_allocator);

    var i: usize = 0;
    while (i < graph.items.len) : (i += 1) {
        // Salta interamente i tripletti ottimizzati
        if (i + 2 < graph.items.len) {
            const pat = try detect_quantized_clip_pattern(graph.items[i .. i + 3]);
            if (pat.detected) {
                i += 2;
                continue;
            }
        }

        const node = graph.items[i];
        for (try node.get_input_tensors()) |tensor| {
            if (tensor.tc != tensorZant_lib.TensorCategory.LINK) continue;

            const name = try tensor.getNameSanitized();
            const canon = canonical(alias_map, name);

            const key = try temp_allocator.dupe(u8, canon);
            const entry = try counts.getOrPut(key);
            if (!entry.found_existing) entry.value_ptr.* = 0;
            entry.value_ptr.* += 1;
        }
    }
    return counts;
}

// Emit reference count decrement for a node's inputs with lazy declaration
fn emitRcDecForNode(
    writer: std.fs.File.Writer,
    node: *NodeZant,
    alias_map: *const std.StringHashMap([]const u8),
    use_counts: *const std.StringHashMap(u32),
    rc_declared: *std.StringHashMap(u8),
    alloc: std.mem.Allocator,
) !void {
    for (try node.get_input_tensors()) |tensor| {
        if (tensor.tc != tensorZant_lib.TensorCategory.LINK) continue;

        const name = try tensor.getNameSanitized();
        const canon = canonical(alias_map, name);

        // Dichiara il contatore solo la prima volta che serve
        if (!rc_declared.contains(canon)) {
            const init = use_counts.get(canon) orelse 0;
            _ = try writer.print("    var _use_count_{s}: u32 = {d};\n", .{ canon, init });
            const key = try alloc.dupe(u8, canon);
            try rc_declared.put(key, 1);
        }

        _ = try writer.print("    // RC dec for {s} (canonical: {s})\n", .{ name, canon });
        _ = try writer.print("    _use_count_{s} -= 1;\n", .{canon});
        _ = try writer.print("    if (_use_count_{s} == 0) tensor_{s}.deinit();\n", .{ canon, canon });
    }
}

// Emit reference count decrement for static mode with FBA pool tracking
fn emitRcDecForNodeStatic(
    writer: std.fs.File.Writer,
    node: *NodeZant,
    alias_map: *const std.StringHashMap([]const u8),
    use_counts: *const std.StringHashMap(u32),
    rc_declared: *std.StringHashMap(u8),
    alloc: std.mem.Allocator,
) !void {
    for (try node.get_input_tensors()) |tensor| {
        if (tensor.tc != tensorZant_lib.TensorCategory.LINK) continue;

        const name = try tensor.getNameSanitized();
        const canon = canonical(alias_map, name);

        // CRITICAL: Don't deallocate clip aliases - they're just pointers to existing tensors
        const is_clip_alias = ((std.mem.startsWith(u8, name, "relu__") or std.mem.startsWith(u8, name, "relu6__")) and std.mem.endsWith(u8, name, "_0_quantized"));
        if (is_clip_alias) continue;

        // Dichiara il contatore solo la prima volta che serve
        if (!rc_declared.contains(canon)) {
            const init = use_counts.get(canon) orelse 0;
            _ = try writer.print("    var _use_count_{s}: u32 = {d};\n", .{ canon, init });
            const key = try alloc.dupe(u8, canon);
            try rc_declared.put(key, 1);
        }

        _ = try writer.print("    // RC dec for {s} (canonical: {s})\n", .{ name, canon });
        _ = try writer.print("    _use_count_{s} -= 1;\n", .{canon});
        _ = try writer.print("    if (_use_count_{s} == 0) {{\n", .{canon});
        _ = try writer.print("        // Static tensor deallocation: decrement FBA pool counter\n", .{});
        _ = try writer.print("        if (tensor_{s}.allocator == &fba_a) {{\n", .{canon});
        _ = try writer.print("            if (fba_live_a > 0) fba_live_a -= 1;\n", .{});
        _ = try writer.print("        }} else {{\n", .{});
        _ = try writer.print("            if (fba_live_b > 0) fba_live_b -= 1;\n", .{});
        _ = try writer.print("        }}\n", .{});
        _ = try writer.print("        // Check if pool is empty and reset if needed\n", .{});
        _ = try writer.print("        if (fba_live_a == 0) {{\n", .{});
        if (codegen_options.log) {
            _ = try writer.print("            if (log_function) |log| {{ log(@constCast(@ptrCast(\"🗑️  FBA Reset(A): pool A empty, reclaiming buffer\\n\"))); }}\n", .{});
        }
        _ = try writer.print("            fba_state_a.reset();\n", .{});
        _ = try writer.print("        }}\n", .{});
        _ = try writer.print("        if (fba_live_b == 0) {{\n", .{});
        if (codegen_options.log) {
            _ = try writer.print("            if (log_function) |log| {{ log(@constCast(@ptrCast(\"🗑️  FBA Reset(B): pool B empty, reclaiming buffer\\n\"))); }}\n", .{});
        }
        _ = try writer.print("            fba_state_b.reset();\n", .{});
        _ = try writer.print("        }}\n", .{});
        _ = try writer.print("    }}\n", .{});
    }
}

//free a linker tensor after dynamical allocation
fn deallocate_useless_link_tensors(writer: std.fs.File.Writer, starting_node: usize, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    const node = linearizedGraph.items[starting_node];

    //After computing the OP, delete link tensors that are not useful anymore when we are in dynamic allocation

    var used: bool = false;
    for (try node.get_input_tensors()) |my_input_tensor| { //for each input tensor of my node
        used = false;

        // AGGRESSIVE DEALLOCATION: Look ahead only 1-2 operations instead of entire graph
        // This reduces peak memory by deallocating tensors sooner
        const lookahead_limit = @min(starting_node + 2, linearizedGraph.items.len);

        //if not used anymore in the next few operations
        for (starting_node + 1..lookahead_limit) |j| {
            for (try linearizedGraph.items[j].get_input_tensors()) |other_input_tens| {
                if (std.mem.eql(u8, my_input_tensor.name, other_input_tens.name)) {
                    used = true;
                    break;
                }
            }
            if (used) break;
        }

        // Use reference counting based deallocation
        if (my_input_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
            const tensor_name = try my_input_tensor.getNameSanitized();

            // CRITICAL: Don't deallocate clip aliases - they're just pointers to existing tensors
            const is_clip_alias = ((std.mem.startsWith(u8, tensor_name, "relu__") or std.mem.startsWith(u8, tensor_name, "relu6__")) and std.mem.endsWith(u8, tensor_name, "_0_quantized"));

            if (!is_clip_alias) {
                // Decrement reference count and deallocate if last use
                _ = try writer.print("    // Reference counting deallocation for {s}\n", .{tensor_name});
                _ = try writer.print("    _use_count_{s} -= 1;\n", .{tensor_name});
                _ = try writer.print("    if (_use_count_{s} == 0) tensor_{s}.deinit();\n", .{ tensor_name, tensor_name });
            }
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

// Structure to hold pattern detection results
const QuantizedClipPatternResult = struct {
    detected: bool,
    dequant_node: ?*NodeZant = null,
    clip_node: ?*NodeZant = null,
    quant_node: ?*NodeZant = null,
    input_quantized_tensor: ?*TensorZant = null,
    input_scale_tensor: ?*TensorZant = null,
    input_zero_point_tensor: ?*TensorZant = null,
    output_quantized_tensor: ?*TensorZant = null,
    output_scale_tensor: ?*TensorZant = null,
    output_zero_point_tensor: ?*TensorZant = null,
    min_val: f32 = 0.0,
    max_val: f32 = 6.0,
};

// Detects the pattern: DequantizeLinear -> Clip -> QuantizeLinear
fn detect_quantized_clip_pattern(nodes: []*NodeZant) !QuantizedClipPatternResult {
    if (nodes.len < 3) return QuantizedClipPatternResult{ .detected = false };

    const node1 = nodes[0];
    const node2 = nodes[1];
    const node3 = nodes[2];

    // Support both Clip and Relu as the middle op
    const is_dequant = std.mem.eql(u8, node1.op_type, "DequantizeLinear");
    const is_quant = std.mem.eql(u8, node3.op_type, "QuantizeLinear");
    const is_clip = std.mem.eql(u8, node2.op_type, "Clip");
    const is_relu = std.mem.eql(u8, node2.op_type, "Relu");

    if (!is_dequant or !is_quant or !(is_clip or is_relu)) {
        return QuantizedClipPatternResult{ .detected = false };
    }

    // Verify tensor connectivity using node-level tensors
    const dequant_outputs = try node1.get_output_tensors();
    const mid_inputs = try node2.get_input_tensors();
    const mid_outputs = try node2.get_output_tensors();
    const quant_inputs = try node3.get_input_tensors();

    if (dequant_outputs.len == 0 or mid_inputs.len == 0) return QuantizedClipPatternResult{ .detected = false };
    if (!std.mem.eql(u8, dequant_outputs[0].name, mid_inputs[0].name)) {
        return QuantizedClipPatternResult{ .detected = false };
    }

    if (mid_outputs.len == 0 or quant_inputs.len == 0) return QuantizedClipPatternResult{ .detected = false };
    if (!std.mem.eql(u8, mid_outputs[0].name, quant_inputs[0].name)) {
        return QuantizedClipPatternResult{ .detected = false };
    }

    // Extract bounds (Clip: read min/max; Relu: [0, +inf))
    var min_val: f32 = 0;
    var max_val: f32 = std.math.floatMax(f32);

    if (is_clip) {
        const clip_op = switch (node2.op) {
            .clip => |op| op,
            else => return QuantizedClipPatternResult{ .detected = false },
        };

        // Force lower bound to 0.0 for quantized in-place optimization semantics
        min_val = 0.0;

        if (clip_op.max) |max_tensor| {
            if (max_tensor.ptr) |tensor_ptr| {
                max_val = tensor_ptr.f32.data[0];
            }
        }
    } else {
        // Relu: lower bound at 0, effectively no upper bound
        min_val = 0.0;
        max_val = std.math.floatMax(f32);
    }

    // Get tensors for the optimized operation (using node-level APIs)
    const dequant_inputs = try node1.get_input_tensors();
    const quant_outputs = try node3.get_output_tensors();
    const quant_inputs_full = try node3.get_input_tensors();

    if (dequant_inputs.len < 3 or quant_outputs.len < 1) return QuantizedClipPatternResult{ .detected = false };
    if (quant_inputs_full.len < 3) return QuantizedClipPatternResult{ .detected = false };

    return QuantizedClipPatternResult{
        .detected = true,
        .dequant_node = node1,
        .clip_node = node2,
        .quant_node = node3,
        .input_quantized_tensor = dequant_inputs[0], // x
        .input_scale_tensor = dequant_inputs[1], // x_scale
        .input_zero_point_tensor = dequant_inputs[2], // x_zero_point
        .output_quantized_tensor = quant_outputs[0], // y
        .output_scale_tensor = quant_inputs_full[1], // y_scale
        .output_zero_point_tensor = quant_inputs_full[2], // y_zero_point
        .min_val = min_val,
        .max_val = max_val,
    };
}

// Writes the optimized quantized clip pattern using clip_quantized_lean
fn write_quantized_clip_pattern(writer: std.fs.File.Writer, pattern: QuantizedClipPatternResult) !void {
    if (!pattern.detected) return;

    if (codegen_options.comm) {
        try writer.print(
            \\
            \\    // OPTIMIZED PATTERN: DequantizeLinear -> Clip -> QuantizeLinear
            \\    // Replaced with direct quantized clip to save memory and computation
            \\    
        , .{});
    }

    if (codegen_options.log) {
        try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Running optimized QuantizedClip (ReLU/Clip) operation...\n")));
            \\    }}
            \\    
        , .{});
    }

    // Note: We don't allocate intermediate tensors (dequant and clip outputs)
    // because the optimization bypasses them entirely using clip_quantized_lean
    // We also don't allocate the output tensor since we're doing in-place clipping

    // Call the optimized clip_quantized_lean function
    try Clip.write_op_quantized_pattern(
        pattern.input_quantized_tensor.?,
        pattern.input_scale_tensor.?,
        pattern.input_zero_point_tensor.?,
        pattern.output_quantized_tensor.?,
        pattern.output_scale_tensor.?,
        pattern.output_zero_point_tensor.?,
        pattern.min_val,
        pattern.max_val,
        writer,
    );

    // Create an alias so subsequent operations can reference the output tensor
    const sanitized_input_name = try pattern.input_quantized_tensor.?.getNameSanitized();
    const sanitized_output_name = try pattern.output_quantized_tensor.?.getNameSanitized();
    try writer.print("    var tensor_{s} = tensor_{s}; // Alias for in-place clip result\n", .{ sanitized_output_name, sanitized_input_name });

    // Note: We don't call deinit() on the input tensor since it's still in use via the alias
    // The alias will be deinit()'ed later when it's no longer needed
}

// Fused Transpose (NHWC f32) -> QuantizeLinear (NCHW u8) pattern
const TransposeQuantizePatternResult = struct {
    detected: bool,
    transpose_node: ?*NodeZant = null,
    quant_node: ?*NodeZant = null,
    input_tensor: ?*TensorZant = null, // NHWC f32 input
    scale_tensor: ?*TensorZant = null,
    zero_point_tensor: ?*TensorZant = null,
    output_tensor: ?*TensorZant = null, // NCHW u8 output
    n: usize = 1,
    h: usize = 0,
    w: usize = 0,
    c: usize = 0,
};

fn is_nhwc_to_nchw(input_shape: []const usize, output_shape: []const usize) bool {
    if (input_shape.len != 4 or output_shape.len != 4) return false;
    const n = input_shape[0];
    const h = input_shape[1];
    const w = input_shape[2];
    const c = input_shape[3];
    return (output_shape[0] == n and output_shape[1] == c and output_shape[2] == h and output_shape[3] == w);
}

fn detect_transpose_quantize_pattern(nodes: []*NodeZant) !TransposeQuantizePatternResult {
    if (nodes.len < 2) return TransposeQuantizePatternResult{ .detected = false };
    const n1 = nodes[0];
    const n2 = nodes[1];

    if (!std.mem.eql(u8, n1.op_type, "Transpose")) return TransposeQuantizePatternResult{ .detected = false };
    if (!std.mem.eql(u8, n2.op_type, "QuantizeLinear")) return TransposeQuantizePatternResult{ .detected = false };

    const t_in = try n1.get_input_tensors();
    const t_out = try n1.get_output_tensors();
    const q_in = try n2.get_input_tensors();
    const q_out = try n2.get_output_tensors();

    if (t_in.len == 0 or t_out.len == 0 or q_in.len < 3 or q_out.len == 0) return TransposeQuantizePatternResult{ .detected = false };

    // Connectivity: transpose output feeds quantize input x
    if (!std.mem.eql(u8, t_out[0].name, q_in[0].name)) return TransposeQuantizePatternResult{ .detected = false };

    // Only fuse when transpose input is external INPUT and is f32, and quantize output is u8
    if (t_in[0].tc != tensorZant_lib.TensorCategory.INPUT) return TransposeQuantizePatternResult{ .detected = false };
    if (t_in[0].ty != tensorZant_lib.TensorType.f32) return TransposeQuantizePatternResult{ .detected = false };
    if (q_out[0].ty != tensorZant_lib.TensorType.u8) return TransposeQuantizePatternResult{ .detected = false };

    // Shape check: NHWC -> NCHW
    const in_shape = t_in[0].getShape();
    const out_shape = q_out[0].getShape();
    if (!is_nhwc_to_nchw(in_shape, out_shape)) return TransposeQuantizePatternResult{ .detected = false };

    return TransposeQuantizePatternResult{
        .detected = true,
        .transpose_node = n1,
        .quant_node = n2,
        .input_tensor = t_in[0],
        .scale_tensor = q_in[1],
        .zero_point_tensor = q_in[2],
        .output_tensor = q_out[0],
        .n = in_shape[0],
        .h = in_shape[1],
        .w = in_shape[2],
        .c = in_shape[3],
    };
}

fn write_transpose_quantize_fused(writer: std.fs.File.Writer, pat: TransposeQuantizePatternResult) !void {
    if (!pat.detected) return;

    const in_name = try pat.input_tensor.?.getNameSanitized();
    const out_name = try pat.output_tensor.?.getNameSanitized();

    // Resolve scale and zero-point expressions
    const scale_is_param = pat.scale_tensor.?.tc == tensorZant_lib.TensorCategory.INITIALIZER;
    const zp_is_param = pat.zero_point_tensor.?.tc == tensorZant_lib.TensorCategory.INITIALIZER;

    const scale_name = try pat.scale_tensor.?.getNameSanitized();
    const zp_name = try pat.zero_point_tensor.?.getNameSanitized();

    // Comment and logging
    if (codegen_options.comm) {
        try writer.print(
            \\
            \\    // OPTIMIZED PATTERN: Transpose (NHWC f32) -> QuantizeLinear (NCHW u8)
            \\    // Fused quantizing transpose to eliminate intermediate f32 buffer
            \\    
        , .{});
    }
    if (codegen_options.log) {
        try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Running fused Transpose+Quantize operation...\n")));
            \\    }}
            \\    
        , .{});
    }

    // Constants for dimensions
    try writer.print("    const N: usize = {d};\n", .{pat.n});
    try writer.print("    const H: usize = {d};\n", .{pat.h});
    try writer.print("    const W: usize = {d};\n", .{pat.w});
    try writer.print("    const C: usize = {d};\n", .{pat.c});

    // Bind input/output data pointers
    try writer.print("    const x = tensor_{s}.data; // NHWC f32\n", .{in_name});
    try writer.print("    const y = tensor_{s}.data; // NCHW u8\n", .{out_name});

    // Load scale and zero point
    if (scale_is_param) {
        try writer.print("    const y_scale: f32 = @constCast(&param_lib.tensor_{s}).data[0];\n", .{scale_name});
    } else {
        try writer.print("    const y_scale: f32 = tensor_{s}.data[0];\n", .{scale_name});
    }
    if (zp_is_param) {
        try writer.print("    const y_zp: {s} = @constCast(&param_lib.tensor_{s}).data[0];\n", .{ pat.zero_point_tensor.?.ty.toString(), zp_name });
    } else {
        try writer.print("    const y_zp: {s} = tensor_{s}.data[0];\n", .{ pat.zero_point_tensor.?.ty.toString(), zp_name });
    }

    // Emit fused loops: iterate N,C,H,W and index NHWC input
    try writer.print(
        \\    var idx: usize = 0;
        \\    var n: usize = 0;
        \\    while (n < N) : (n += 1) {{
        \\        var c: usize = 0;
        \\        while (c < C) : (c += 1) {{
        \\            var h: usize = 0;
        \\            while (h < H) : (h += 1) {{
        \\                var w: usize = 0;
        \\                while (w < W) : (w += 1) {{
        \\                    const nhwc_index = (((n * H) + h) * W + w) * C + c;
        \\                    const v: f32 = x[nhwc_index];
        \\                    const q_i32: i32 = @as(i32, @intFromFloat(@round(v / y_scale))) + @as(i32, y_zp);
        \\                    const q_clamped: i32 = @max(0, @min(255, q_i32));
        \\                    y[idx] = @intCast(q_clamped);
        \\                    idx += 1;
        \\                }}
        \\            }}
        \\        }}
        \\    }}
    , .{});
}

// QAdd fusion pattern: two dequantized inputs added then re-quantized
const QAddPatternResult = struct {
    detected: bool,
    deq_a_node: ?*NodeZant = null,
    deq_b_node: ?*NodeZant = null,
    add_node: ?*NodeZant = null,
    quant_node: ?*NodeZant = null,
    a_q: ?*TensorZant = null,
    a_scale: ?*TensorZant = null,
    a_zp: ?*TensorZant = null,
    b_q: ?*TensorZant = null,
    b_scale: ?*TensorZant = null,
    b_zp: ?*TensorZant = null,
    c_q: ?*TensorZant = null,
    c_scale: ?*TensorZant = null,
    c_zp: ?*TensorZant = null,
};

fn detect_qadd_pattern(nodes: []*NodeZant) !QAddPatternResult {
    if (nodes.len < 4) return QAddPatternResult{ .detected = false };
    const n0 = nodes[0];
    const n1 = nodes[1];
    const n2 = nodes[2];
    const n3 = nodes[3];

    if (!std.mem.eql(u8, n0.op_type, "DequantizeLinear")) return QAddPatternResult{ .detected = false };
    if (!std.mem.eql(u8, n1.op_type, "DequantizeLinear")) return QAddPatternResult{ .detected = false };
    if (!std.mem.eql(u8, n2.op_type, "Add")) return QAddPatternResult{ .detected = false };
    if (!std.mem.eql(u8, n3.op_type, "QuantizeLinear")) return QAddPatternResult{ .detected = false };

    const deq0_out = try n0.get_output_tensors();
    const deq1_out = try n1.get_output_tensors();
    const add_in = try n2.get_input_tensors();
    const add_out = try n2.get_output_tensors();
    const q_in = try n3.get_input_tensors();
    const q_out = try n3.get_output_tensors();

    if (deq0_out.len == 0 or deq1_out.len == 0 or add_in.len < 2 or add_out.len == 0 or q_in.len < 3 or q_out.len == 0) {
        return QAddPatternResult{ .detected = false };
    }

    // Check that add inputs are the dequant outputs (order-free)
    const d0 = deq0_out[0].name;
    const d1 = deq1_out[0].name;
    const a0 = add_in[0].name;
    const a1 = add_in[1].name;
    const add_inputs_match = (std.mem.eql(u8, d0, a0) and std.mem.eql(u8, d1, a1)) or (std.mem.eql(u8, d0, a1) and std.mem.eql(u8, d1, a0));
    if (!add_inputs_match) return QAddPatternResult{ .detected = false };

    // Add output feeds QuantizeLinear input x
    if (!std.mem.eql(u8, add_out[0].name, q_in[0].name)) return QAddPatternResult{ .detected = false };

    // Gather input quantized tensors and scales/zps
    const deq0_in = try n0.get_input_tensors();
    const deq1_in = try n1.get_input_tensors();
    if (deq0_in.len < 3 or deq1_in.len < 3) return QAddPatternResult{ .detected = false };

    return QAddPatternResult{
        .detected = true,
        .deq_a_node = n0,
        .deq_b_node = n1,
        .add_node = n2,
        .quant_node = n3,
        .a_q = deq0_in[0],
        .a_scale = deq0_in[1],
        .a_zp = deq0_in[2],
        .b_q = deq1_in[0],
        .b_scale = deq1_in[1],
        .b_zp = deq1_in[2],
        .c_q = q_out[0],
        .c_scale = q_in[1],
        .c_zp = q_in[2],
    };
}

fn write_qadd_fused(writer: std.fs.File.Writer, pat: QAddPatternResult) !void {
    if (!pat.detected) return;

    // Helper to create tensor reference strings similar to other fused writers
    const makeRef = struct {
        fn call(t: *TensorZant) ![]u8 {
            if (t.tc == tensorZant_lib.TensorCategory.INITIALIZER) {
                return try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try t.getNameSanitized(),
                    ")",
                });
            } else {
                return try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&tensor_",
                    try t.getNameSanitized(),
                    ")",
                });
            }
        }
    }.call;

    const a_q_ref = try makeRef(pat.a_q.?);
    defer allocator.free(a_q_ref);
    const a_scale_ref = try makeRef(pat.a_scale.?);
    defer allocator.free(a_scale_ref);
    const a_zp_ref = try makeRef(pat.a_zp.?);
    defer allocator.free(a_zp_ref);
    const b_q_ref = try makeRef(pat.b_q.?);
    defer allocator.free(b_q_ref);
    const b_scale_ref = try makeRef(pat.b_scale.?);
    defer allocator.free(b_scale_ref);
    const b_zp_ref = try makeRef(pat.b_zp.?);
    defer allocator.free(b_zp_ref);
    const c_scale_ref = try makeRef(pat.c_scale.?);
    defer allocator.free(c_scale_ref);
    const c_zp_ref = try makeRef(pat.c_zp.?);
    defer allocator.free(c_zp_ref);

    const out_name = try pat.c_q.?.getNameSanitized();

    if (codegen_options.comm) {
        try writer.print(
            \\
            \\    // OPTIMIZED PATTERN: DequantizeLinear + DequantizeLinear + Add + QuantizeLinear
            \\    // Replaced with direct qlinearadd_lean to save passes and allocations
            \\    
        , .{});
    }
    if (codegen_options.log) {
        try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Running fused QLinearAdd operation...\n")));
            \\    }}
            \\    
        , .{});
    }

    // Emit the fused call
    try writer.print("    tensMath.qlinearadd_lean(\n", .{});
    try writer.print("        {s},\n", .{a_q_ref});
    try writer.print("        {s},\n", .{a_scale_ref});
    try writer.print("        {s},\n", .{a_zp_ref});
    try writer.print("        {s},\n", .{b_q_ref});
    try writer.print("        {s},\n", .{b_scale_ref});
    try writer.print("        {s},\n", .{b_zp_ref});
    try writer.print("        &tensor_{s},\n", .{out_name});
    try writer.print("        {s},\n", .{c_scale_ref});
    try writer.print("        {s},\n", .{c_zp_ref});
    try writer.print("    ) catch return -1;\n", .{});
}
