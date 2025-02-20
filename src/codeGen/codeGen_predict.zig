const std = @import("std");
const Tensor = @import("tensor").Tensor;
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;
const NodeProto = @import("onnx").NodeProto;
const GraphProto = @import("onnx").GraphProto;
const AttributeProto = @import("onnx").AttributeProto;
const allocator = @import("pkgAllocator").allocator;

const utils = @import("codeGen_utils.zig");
const mathGen = @import("codeGen_math_handler.zig");

var readyGraph: std.ArrayList(ReadyNode) = std.ArrayList(ReadyNode).init(allocator);
var tensorHashMap: std.StringHashMap(ReadyTensor) = std.StringHashMap(ReadyTensor).init(allocator); //key: TensorProto.name

var networkInput: []const u8 = undefined;
var networkOutput: []const u8 = undefined;

// Struct to represent a tensor that is ready for computation
pub const ReadyTensor = struct {
    name: []const u8,
    ready: bool,
    shape: []const i64,

    // Creates a ReadyTensor by checking if it exists in the initializers or is an input
    pub fn create(name: []const u8, graph: *GraphProto) !ReadyTensor {
        //std.debug.print("\n     ReadyTensor.create() --> {s}\n", .{name});
        // const shape_array: []const i64 = &[_]i64{ 1, 1, 1, 1 }; // Placeholder shape //TODO: infer the shape based on the inputs and the op_type

        //TODO: given the operation type and the shape of the input return the shape of the output
        if (utils.isInitializer(name, graph.initializers)) { // Check if tensor is an initializer
            const init: *TensorProto = try utils.getInitializer(name, graph.initializers);
            return ReadyTensor{
                .name = name,
                .ready = true,
                .shape = init.dims,
            };
        } else if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "input")) |_| {
            networkInput = name;
            return ReadyTensor{ // Check if tensor is an input
                .name = name,
                .ready = true,
                .shape = &[_]i64{ 1, 1, 28, 28 }, //TODO; hardcoded shit, ask to Mirko
            };
        } else if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "images")) |_| return ReadyTensor{ // Check if tensor is images
            .name = name,
            .ready = true,
            .shape = &[_]i64{ 1, 1, 1, 1 }, // The real shape is computed during the creation of the node
        } else if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "constant")) |_| return ReadyTensor{
            .name = name,
            .ready = true,
            .shape = &[_]i64{ 1, 1, 1, 1 }, // it will be changed in the graph gration
        } else {
            if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "output")) |_| networkOutput = name;
            return ReadyTensor{ //default
                .name = name,
                .ready = false,
                .shape = &[_]i64{ 1, 1, 1, 1 }, //TODO: given the operation type (op_type) and the shape of the input return the shape of the output
            };
        }
    }
};

// Struct representing a computational node in the ONNX model
pub const ReadyNode = struct {
    nodeProto: *NodeProto,
    inputs: std.ArrayList(*ReadyTensor),
    outputs: std.ArrayList(*ReadyTensor),
    ready: bool,

    // Creates a ReadyNode by preparing its input and output tensors
    pub fn create(nodeProto: *NodeProto) !ReadyNode {
        // std.debug.print("\n\nReadyNode.create() --> {s}", .{nodeProto.name.?});
        var newReadyNode = ReadyNode{
            .nodeProto = nodeProto,
            .inputs = std.ArrayList(*ReadyTensor).init(allocator),
            .outputs = std.ArrayList(*ReadyTensor).init(allocator),
            .ready = false,
        };

        // var inputs = std.ArrayList(*ReadyTensor).init(allocator);
        // var outputs = std.ArrayList(*ReadyTensor).init(allocator);

        for (nodeProto.input) |input_name| { //for each input tensor in NodeProto

            //adding the readyTensor to the model
            try newReadyNode.inputs.append(if (tensorHashMap.getPtr(input_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // std.debug.print("\n   added input {s} to node {s} ", .{ input_name, nodeProto.name.? });

        }
        for (nodeProto.output) |output_name| { //for each output tensor

            //adding the readyTensor to the model
            try newReadyNode.outputs.append(if (tensorHashMap.getPtr(output_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // std.debug.print("\n   added output {s} to node {s} ", .{ output_name, nodeProto.name.? });
        }

        // -- COMPUTING THE OUTPUT SHAPE --
        try mathGen.compute_output_shape(&newReadyNode);

        return newReadyNode;
    }
};

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: std.fs.File.Writer, model: ModelOnnx) !void {

    //create the hashMap
    try createReadyTensorHashMap(model);

    //DEBUG
    //utils.printTensorHashMap(tensorHashMap);

    //DEBUG
    try utils.printOperations(model.graph.?);

    //create the ReadyGraph
    try createReadyGraph(model);

    //DEBUG
    try utils.printNodeList(readyGraph);

    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         initializing output Tensors             +
        \\ // ---------------------------------------------------
    , .{});

    try writeInitOutputs(writer);

    _ = try writer.print(
        \\
        \\
        \\
        \\fn predict(T: anytype, input: [*]T, input_shape: []usize) ![*]T {{
    , .{});

    try writeInitInput(writer);

    try writeComputationGraph(writer);

    try writeReturn(writer);

    std.debug.print("\n#############################################################", .{});
    std.debug.print("\n+                      EXECUTION ENDED                      +", .{});
    std.debug.print("\n#############################################################", .{});
}

// ----------------------- HASH MAP -----------------------
// Populates tensorHashMap with the tensors used in the onn graph, where the key is the name of the tensor
fn createReadyTensorHashMap(model: ModelOnnx) !void {
    const protoGraph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto
        for (node.input) |input_name| {
            try addToTensorHashMap(input_name, model.graph.?);
        }
        for (node.output) |output_name| {
            try addToTensorHashMap(output_name, model.graph.?);
        }
    }
}

inline fn addToTensorHashMap(name: []const u8, graph: *GraphProto) !void {
    if (tensorHashMap.get(name)) |_| {
        // std.debug.print("\n ----- Tensor {s} already present!! ", .{name});
        return;
    } else {
        //create the readyTensor
        const readyTensor: ReadyTensor = try ReadyTensor.create(name, graph);
        //add the readyTensor to the HashMap
        try tensorHashMap.put(name, readyTensor);
        // std.debug.print("\n +++++ Tensor {s} added to the hash map ", .{name});
    }
}

// ----------------------- READY GRAPH -----------------------

// Creates a graph representation with all nodes in a ready-to-compute state
fn createReadyGraph(model: ModelOnnx) !void {
    const graph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    for (graph.nodes) |node_ptr| { //for each NodeProto in the GraphProto

        const readyNode = try ReadyNode.create(node_ptr);
        try readyGraph.append(readyNode);
    }
}

// Processes and writes the computation graph
inline fn writeComputationGraph(writer: std.fs.File.Writer) !void {
    var iteration: usize = 0;

    while (true) {
        // DEBUG
        // std.debug.print("\n\n=== Iteration {} ===\n", .{iteration});
        // // Print status of all nodes
        // for (readyGraph.items) |*node| {
        //     std.debug.print("Node {s} (op: {s}): {s}\n", .{
        //         node.nodeProto.name orelse "unnamed",
        //         node.nodeProto.op_type,
        //         if (node.ready) "COMPLETED" else if (utils.areAllInputsReady(node)) "READY" else "WAITING",
        //     });
        //     // Print input tensor status
        //     for (node.inputs.items) |input| {
        //         std.debug.print("  Input {s}: {s}\n", .{ input.name, if (input.ready) "ready" else "not ready" });
        //     }
        // }

        const computableNodes: std.ArrayList(*ReadyNode) = try utils.getComputableNodes(&readyGraph);
        //DEBUG
        //try utils.printComputableNodes(computableNodes);

        if (computableNodes.items.len == 0) return;

        for (computableNodes.items) |node_ptr| {
            //writing the operation
            try writeOperation(writer, node_ptr);
            //set the output as ready
            try utils.setOutputsReady(node_ptr, &tensorHashMap);
        }
        iteration += 1;
    }
}

// Initializes output tensors in the computation graph
fn writeInitOutputs(writer: std.fs.File.Writer) !void {
    for (readyGraph.items) |*node| {
        //writing the outputs, OSS: two nodes shpuld never have the same output by definition, so we don't need to check for duplicates
        for (node.outputs.items) |output| {
            if (std.mem.eql(u8, node.nodeProto.op_type, "Constant") and node.inputs.items.len == 0) { //A node is constant if it only has one output and no inputs
                if (node.outputs.items.len > 1) return error.MultipleOutputConstant else {
                    try writeConstant(writer, node);
                    //set the node and tensor to Ready
                    var mutableNode: *ReadyNode = @constCast(node);
                    mutableNode.ready = true;
                }
            } else {
                try writeOutputShape(
                    writer,
                    output,
                );
                try writeOutputTensor(
                    writer,
                    output.name,
                );
            }
        }
    }
}

fn writeOutputShape(writer: std.fs.File.Writer, output: *ReadyTensor) !void {
    const shape = output.shape;
    try writer.print(
        \\
        \\
        \\const shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{
        try utils.getSanitizedName(output.name),
        shape.len,
    });

    for (0..shape.len) |i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print(
            \\ {}
        , .{shape[i]});
    }

    try writer.print(
        \\}} ;
    , .{});
}

fn writeConstant(writer: std.fs.File.Writer, readyNode: *const ReadyNode) !void {
    try writer.print(
        \\
        \\ // ---- CONSTANT TENSOR ---- readyNode.nodeProto.attribute: {any}
    , .{readyNode.nodeProto.attribute});

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

    const dataTypeString = try utils.getTypeString(tensor.data_type);
    try writer.print(
        \\
        \\const array_{s} : [{d}]{s} = [_]{s}{{
    , .{ sanitized_name, total_size, dataTypeString, dataTypeString });

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
                const float_data = @as([*]const f32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
                for (0..float_data.len) |i| {
                    if (i > 0) try writer.print(",", .{});
                    try writer.print(" {d}", .{float_data[i]});
                }
            },
            .INT64 => {
                const int_data = @as([*]const i64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
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
        \\const tensor_{s} = Tensor({s}).fromArray(&allocator, &array_{s}, &shape_tensor_{s});
    , .{ sanitized_name, dataTypeString, sanitized_name, sanitized_name });
}

fn writeOutputTensor(writer: std.fs.File.Writer, name: []const u8) !void {
    const sanitized_name = try utils.getSanitizedName(name);
    try writer.print(
        \\
        \\const tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s});
    , .{ sanitized_name, "f32", sanitized_name });
}

fn writeInitInput(writer: std.fs.File.Writer) !void {

    //compute the size:
    _ = try writer.print(
        \\  
        \\     if (input_shape.len == 0) return error.ShapeLenZero;
        \\     var size: u16 = 1;
        \\     for(input_shape) |dim_i| {{
        \\         size *= dim_i;
        \\     }}
        \\
        \\     const data = allocator.alloc(T, size) catch return null;
        \\  
        \\     for (0..size) |i| {{
        \\         data[i] = input[i]; // Copying input elements 
        \\     }}
        \\
        \\     var tensor_{s} = Tensor(T).fromShape(&allocator, &input_shape);
        \\
    , .{try utils.getSanitizedName(networkInput)});
}

fn writeOperation(writer: std.fs.File.Writer, readyNode: *ReadyNode) !void {
    try mathGen.write_math_op(writer, readyNode);
}

fn writeReturn(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\      return &tensor_{s}.data ;
        \\}}
    , .{networkOutput});
}
