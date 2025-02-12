const std = @import("std");
const Tensor = @import("tensor").Tensor;
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;
const NodeProto = @import("onnx").NodeProto;
const GraphProto = @import("onnx").GraphProto;
const allocator = @import("pkgAllocator").allocator;

const codeGenInitializers = @import("codeGen_initializers.zig");
const utils = @import("codeGen_utils.zig");
const mathGen = @import("codeGen_math_handler.zig");

var readyGraph: std.ArrayList(ReadyNode) = std.ArrayList(ReadyNode).init(allocator);
var tensorHashMap: std.StringHashMap(ReadyTensor) = std.StringHashMap(ReadyTensor).init(allocator); //key: TensorProto.name

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
        } else if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "input")) |_| return ReadyTensor{ // Check if tensor is an input
            .name = "input",
            .ready = true,
            .shape = &[_]i64{ 1, 1, 1, 1 }, //TODO: given the operation type (op_type) and the shape of the input return the shape of the output
        } else return ReadyTensor{
            .name = name,
            .ready = false,
            .shape = &[_]i64{ 1, 1, 1, 1 }, //TODO: given the operation type (op_type) and the shape of the input return the shape of the output
        };
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

        var inputs = std.ArrayList(*ReadyTensor).init(allocator);
        var outputs = std.ArrayList(*ReadyTensor).init(allocator);
        for (nodeProto.input) |input_name| { //for each input tensor in NodeProto

            //adding the readyTensor to the model
            try inputs.append(if (tensorHashMap.getPtr(input_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // std.debug.print("\n   added input {s} to node {s} ", .{ input_name, nodeProto.name.? });
        }
        for (nodeProto.output) |output_name| { //for each output tensor

            //adding the readyTensor to the model
            try outputs.append(if (tensorHashMap.getPtr(output_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // std.debug.print("\n   added output {s} to node {s} ", .{ output_name, nodeProto.name.? });
        }

        return ReadyNode{
            .nodeProto = nodeProto,
            .inputs = inputs,
            .outputs = outputs,
            .ready = false,
        };
    }
};

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: std.fs.File.Writer, model: ModelOnnx) !void {

    //create the hashMap
    try createReadyTensorHashMap(model);
    //DEBUG
    std.debug.print("\n-------------------------------------------------------------", .{});
    std.debug.print("\n+                       READY HASHMAP                       +", .{});
    std.debug.print("\n-------------------------------------------------------------", .{});
    //DEBUG
    // utils.printTensorHashMap(tensorHashMap);

    //create the ReadyGraph
    try createReadyGraph(model);
    //DEBUG
    std.debug.print("\n-------------------------------------------------------------", .{});
    std.debug.print("\n+                        READY GRAPH                        +", .{});
    std.debug.print("\n-------------------------------------------------------------", .{});
    //DEBUG
    // try utils.printNodeList(readyGraph);

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
        \\pub fn predict(comptime T: anytype, tensor_input: Tensor(T)) !void {{
    , .{});

    try writeComputationGraph(writer);

    _ = try writer.print(
        \\
        \\ }}
    , .{});

    std.debug.print("\n#############################################################", .{});
    std.debug.print("\n+                      EXECUTION ENDED                      +", .{});
    std.debug.print("\n#############################################################", .{});
}

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

    //Create a list with all the nodes ready for the computation
    //A node is ready for the computation when all its input tensors are ready

    while (true) { //TODO: while(true)==desperation, do something more elegant

        const computableNodes: std.ArrayList(*ReadyNode) = try utils.getComputableNodes(&readyGraph);
        //DEBUG
        std.debug.print("\n------------------------------------------------------------", .{});
        std.debug.print("\n+                  COMPUTABLE NODES  n:{}                  +", .{computableNodes.items.len});
        std.debug.print("\n------------------------------------------------------------", .{});
        try utils.printComputableNodes(computableNodes);

        if (computableNodes.items.len == 0) return;

        for (computableNodes.items) |node_ptr| {

            //writing the operation
            try writeOperation(writer, node_ptr);

            //set the output as ready
            try setOutputsReady(node_ptr);
        }
    }
}

// Initializes output tensors in the computation graph
fn writeInitOutputs(writer: std.fs.File.Writer) !void {
    for (readyGraph.items) |node_ptr| {
        //writing the outputs, OSS: two nodes shpuld never have the same output, so we don't need to check for duplicates
        for (node_ptr.outputs.items) |output| {
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

    try writer.print(
        \\{}
    , .{shape[0]});
    for (1..shape.len) |i| {
        try writer.print(
            \\, {}
        , .{shape[i]});
    }

    try writer.print(
        \\}} ;
    , .{});
}

fn writeOutputTensor(writer: std.fs.File.Writer, name: []const u8) !void {
    const sanitized_name = try utils.getSanitizedName(name);
    try writer.print(
        \\
        \\const tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s});
    , .{ sanitized_name, "f32", sanitized_name });
}

fn writeOperation(writer: std.fs.File.Writer, readyNode: *ReadyNode) !void {
    try writer.print(
        \\
        \\
        \\    //forwarding operation : {s}
        \\    //parameters:
        \\    //   inputs: 
    , .{readyNode.*.nodeProto.*.op_type});

    //write the inputs
    for (readyNode.inputs.items) |input| {
        try writer.print(
            \\
            \\    //      -> {s} 
        , .{input.name});
    }
    try writer.print(
        \\
        \\    //    outputs: 
    , .{});

    //write the outputs
    for (readyNode.outputs.items) |output| {
        try writer.print(
            \\
            \\    //      <- {s} 
        , .{output.name});
    }

    try mathGen.write_math_op(writer, readyNode);
}

// Marks output tensors as ready for computation in all the graph
fn setOutputsReady(completedNode: *ReadyNode) !void {
    std.debug.print("\n -----> set {s} outputs to ready", .{completedNode.nodeProto.name.?});
    completedNode.ready = true;
    for (completedNode.outputs.items) |ready_output_tensor| { //for each output tensor of the completed node
        var mutablePtr: *ReadyTensor = if (tensorHashMap.getPtr(ready_output_tensor.name)) |V_ptr| V_ptr else return error.keyNotAvailable;
        mutablePtr.ready = true;
        std.debug.print("\n    {s} --> ready", .{mutablePtr.name});
    }
}
