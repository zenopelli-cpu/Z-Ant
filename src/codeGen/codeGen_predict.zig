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

pub const ReadyTensor = struct {
    name: []const u8,
    ready: bool,
    shape: []const i64,

    pub fn create(name: []const u8, graph: *GraphProto, op_type: []const u8) !ReadyTensor {
        std.debug.print("\n     ReadyTensor.create() --> {s}\n", .{name});
        _ = op_type;
        const shape_array: []const i64 = &[_]i64{ 1, 1, 1, 1 }; //TODO: given the operation type (op_type) and the shape of the input return the shape of the output

        //TODO: given the operation type and the shape of the input return the shape of the output
        if (utils.isInitializer(name, graph.initializers)) {
            const init: *TensorProto = try utils.getInitializer(name, graph.initializers);
            return ReadyTensor{
                .name = name,
                .ready = true,
                .shape = init.dims,
            };
        } else if (std.mem.indexOf(u8, try utils.getSanitizedName(name), "input")) |_| return ReadyTensor{
            .name = "input",
            .ready = true,
            .shape = shape_array, //TODO: given the operation type (op_type) and the shape of the input return the shape of the output
        } else return ReadyTensor{
            .name = name,
            .ready = false,
            .shape = shape_array, //TODO: given the operation type (op_type) and the shape of the input return the shape of the output
        };
    }
};

pub const ReadyNode = struct {
    nodeProto: *NodeProto,
    inputs: std.ArrayList(ReadyTensor),
    outputs: std.ArrayList(ReadyTensor),
    ready: bool,

    pub fn create(nodeProto: *NodeProto, model: *const ModelOnnx) !ReadyNode {
        std.debug.print("\n ReadyNode.create() --> {s}\n", .{nodeProto.name.?});

        var ready_node = ReadyNode{
            .nodeProto = nodeProto,
            .inputs = std.ArrayList(ReadyTensor).init(allocator),
            .outputs = std.ArrayList(ReadyTensor).init(allocator),
            .ready = false,
        };

        for (nodeProto.input) |input_name| { //for each input tensor in NodeProto

            //creating the ReadyTensor
            const readyInput = try ReadyTensor.create(input_name, model.graph.?, nodeProto.op_type);
            //adding the readyTensor to the model
            try ready_node.inputs.append(readyInput);
        }

        for (nodeProto.output) |output_name| { //for each output tensor
            //creating the ReadyTensor
            const readyOutput = try ReadyTensor.create(output_name, model.graph.?, nodeProto.op_type);
            //adding the readyTensor to the model
            try ready_node.outputs.append(readyOutput);
        }
        return ready_node;
    }
};

pub inline fn writePredict(writer: std.fs.File.Writer, model: ModelOnnx) !void {

    //create the ReadyGraph
    var readyGraph: std.ArrayList(ReadyNode) = try createReadyGraph(model);
    _ = &readyGraph;

    //DEBUG
    std.debug.print("\n------------------------------------------------------------", .{});
    std.debug.print("\n+                READY GRAPH after Creation                +", .{});
    std.debug.print("\n------------------------------------------------------------", .{});
    try utils.printNodeList(readyGraph);

    _ = try writer.print(
        \\
        \\
        \\
        \\pub fn predict(comptime T: anytype, tensor_input: Tensor(T)) !void {{
    , .{});

    while (readyGraph.items.len != 0) {
        try writeComputationGraph(writer, &readyGraph);
    }

    _ = try writer.print(
        \\
        \\}}
    , .{});
}

fn createReadyGraph(model: ModelOnnx) !std.ArrayList(ReadyNode) {
    const graph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    var readyGraph = std.ArrayList(ReadyNode).init(allocator);
    for (graph.nodes) |node_ptr| { //for each NodeProto in the GraphProto

        const readyNode = try ReadyNode.create(node_ptr, &model);
        try readyGraph.append(readyNode);
    }

    return readyGraph;
}

inline fn writeComputationGraph(writer: std.fs.File.Writer, readyGraph: *std.ArrayList(ReadyNode)) !void {

    //Create a list with all the nodes ready for the computation
    //A node is ready for the computation when all its input tensors are ready
    const computableNodes: std.ArrayList(*ReadyNode) = try utils.getComputableNodes(readyGraph);
    //DEBUG
    std.debug.print("\n", .{});
    std.debug.print("\n------------------------------------------------------------", .{});
    std.debug.print("\n+  COMPUTABLE NODES (aka nodes with all the inputs ready)  +", .{});
    std.debug.print("\n------------------------------------------------------------", .{});
    try utils.printComputableNodes(computableNodes);

    for (computableNodes.items) |node_ptr| {
        //writing the outputs
        for (node_ptr.outputs.items) |*output| {
            try writeOutputShape(
                writer,
                output,
            );
            try writeOutputTensor(
                writer,
                output.name,
            );
        }

        //writing the operation
        try writeOperation(writer, node_ptr);

        //set the output as ready
        try setOutputsReady(node_ptr, readyGraph);
    }

    //important step: Delete the computed node from the list
    try removeCompletedNodes(readyGraph);
}

fn writeOutputShape(writer: std.fs.File.Writer, output: *ReadyTensor) !void {
    try writer.print(
        \\
        \\
        \\  const shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{
        try utils.getSanitizedName(output.name),
        output.shape.len,
    });

    try writer.print(
        \\{}
    , .{output.shape[0]});
    for (1..output.shape.len) |i| {
        try writer.print(
            \\, {}
        , .{output.shape[i]});
    }

    try writer.print(
        \\}} ;
    , .{});
}

fn writeOutputTensor(writer: std.fs.File.Writer, name: []const u8) !void {
    const sanitized_name = try utils.getSanitizedName(name);
    try writer.print(
        \\
        \\  const tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s});
    , .{ sanitized_name, "f32", sanitized_name });
}

fn writeOperation(writer: std.fs.File.Writer, readyNode: *ReadyNode) !void {
    try writer.print(
        \\
        \\  //forwarding operation : {s}
        \\  //parameters:
        \\  //   inputs: 
    , .{readyNode.nodeProto.op_type});

    //write the inputs
    for (readyNode.inputs.items) |input| {
        try writer.print(
            \\
            \\  //      -> {s} 
        , .{input.name});
    }

    try writer.print(
        \\
        \\  //    outputs: 
    , .{});
    //write the inputs
    for (readyNode.outputs.items) |output| {
        try writer.print(
            \\
            \\  //      <- {s} 
        , .{output.name});
    }
}

fn setOutputsReady(completedNode: *ReadyNode, readyGraph: *std.ArrayList(ReadyNode)) !void {
    for (completedNode.outputs.items) |*ready_output_tensor| { //for each output tensor of the completed node
        ready_output_tensor.ready = true;
        for (readyGraph.items) |ready_node| { //for each node of the graph
            for (ready_node.inputs.items) |*input| { //for each Tensor of the ready_node's inputs
                if (std.mem.eql(u8, input.name, ready_output_tensor.name)) {
                    input.ready = true;
                }
            }
        }
    }
}

fn removeCompletedNodes(readyGraph: *std.ArrayList(ReadyNode)) !void {
    var i: usize = 0;
    while (i < readyGraph.items.len) {
        if (try utils.isComputed(&readyGraph.items[i])) {
            // remove the node
            std.debug.print("\nremoving node: {s}", .{readyGraph.items[i].nodeProto.name.?});

            _ = readyGraph.swapRemove(i);
            // Do not increment i because swapRemove swaps the element, and we need to check the new element at index i.
        } else {
            i += 1; // Only increment i if we don't remove the node
        }
    }
}
