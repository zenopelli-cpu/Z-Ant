const std = @import("std");
const DataType = @import("onnx").DataType;
const GraphProto = @import("onnx").GraphProto;
const NodeProto = @import("onnx").NodeProto;
const TensorProto = @import("onnx").TensorProto;
const allocator = @import("pkgAllocator").allocator;
const ReadyNode = @import("codeGen_predict.zig").ReadyNode;
const ReadyTensor = @import("codeGen_predict.zig").ReadyTensor;

// -------------------- GETTERS --------------------

//Given an element from DataType Enum in onnx.zig returns the equivalent zig type
pub inline fn getType(data_type: DataType) !type {
    switch (data_type) {
        .FLOAT => {
            return f32;
        },
        .UINT8 => {
            return u8;
        },
        .INT8 => {
            return i8;
        },
        .UINT16 => {
            return u16;
        },
        .INT16 => {
            return i16;
        },
        .INT32 => {
            return i32;
        },
        .INT64 => {
            return i64;
        },
        .FLOAT16 => {
            return f16;
        },
        .DOUBLE => {
            return f64;
        },
        .UNIT32 => {
            return u32;
        },
        .UINT64 => {
            return u64;
        },
        else => return error.DataTypeNotAvailable,
    }
}

//Given an element from DataType Enum in onnx.zig returns the equivalent string of a zig type
pub inline fn getTypeString(data_type: DataType) ![]const u8 {
    switch (data_type) {
        .FLOAT => {
            return "f32";
        },
        .UINT8 => {
            return "u8";
        },
        .INT8 => {
            return "i8";
        },
        .UINT16 => {
            return "u16";
        },
        .INT16 => {
            return "i16";
        },
        .INT32 => {
            return "i32";
        },
        .INT64 => {
            return "i64";
        },
        .FLOAT16 => {
            return "f16";
        },
        .DOUBLE => {
            return "f64";
        },
        .UINT32 => {
            return "u32";
        },
        .UINT64 => {
            return "u64";
        },
        else => return error.DataTypeNotAvailable,
    }
}

//Returns the sanitized tensor's name, removes all non alphanumeric chars
pub inline fn getSanitizedName(name: []const u8) ![]const u8 {
    var sanitized = try allocator.alloc(u8, name.len);

    for (name, 0..) |char, i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(char) or char == '_')
            std.ascii.toLower(char)
        else
            '_';
    }

    //std.debug.print("\nfrom {s} to {s} ", .{ name, sanitized });

    return sanitized;
}

/// Returns a List of Ready nodes
/// A node is considered "computable" if all the node's input Tensors are set as ready
pub inline fn getComputableNodes(readyGraph: *std.ArrayList(ReadyNode)) !std.ArrayList(*ReadyNode) {
    std.debug.print("\n\n getComputableNodes()", .{});

    var set: std.ArrayList(*ReadyNode) = std.ArrayList(*ReadyNode).init(allocator);
    var ready_input_counter: i8 = 0;

    for (readyGraph.items) |*node| {
        if (!node.ready) {
            for (node.inputs.items) |input| {
                if (input.ready) ready_input_counter += 1;
            }
            for (node.outputs.items) |output| {
                if (output.ready) return error.OutputReadyTooEarly;
            }
            if (ready_input_counter == node.inputs.items.len) {
                try set.append(node);
                std.debug.print("\n    --- {s} is computable", .{node.nodeProto.name.?});
            }
            ready_input_counter = 0;
        }
    }

    return set;
}

pub inline fn getConstantTensorDims(nodeProto: *NodeProto) ![]const i64 {
    //check the node is a Constant
    if (std.mem.indexOf(u8, try getSanitizedName(nodeProto.op_type), "constant")) |_| {} else return error.NodeNotConstant;

    return if (nodeProto.attribute[0].t) |tensorProto| tensorProto.dims else error.ConstantTensorAttributeNotAvailable;
}

// -------------------- SETTERS --------------------

// Marks output tensors as ready for computation in all the graph
pub fn setOutputsReady(completedNode: *ReadyNode, tensorHashMap: *std.StringHashMap(ReadyTensor)) !void {
    std.debug.print("\n -----> set {s} outputs to ready", .{completedNode.nodeProto.name.?});
    completedNode.ready = true;
    for (completedNode.outputs.items) |ready_output_tensor| { //for each output tensor of the completed node
        var mutablePtr: *ReadyTensor = if (tensorHashMap.getPtr(ready_output_tensor.name)) |V_ptr| V_ptr else return error.keyNotAvailable;
        mutablePtr.ready = true;
        std.debug.print("\n    {s} --> ready", .{mutablePtr.name});
    }
}

// -------------------- BOOLEANS --------------------

// returns true if all the inputs are ready
pub inline fn areAllInputsReady(node: *ReadyNode) bool {
    for (node.inputs.items) |input| {
        if (!input.ready) return false;
    }
    return true;
}

//returns true if all the inputs and all the outputs of a node are set as ready
pub inline fn isComputed(readyNode: *ReadyNode) !bool {
    for (readyNode.inputs.items) |input| {
        if (!input.ready) return false;
    }
    for (readyNode.outputs.items) |output| {
        if (!output.ready) return false;
    }
    return true;
}

//return true if the first parameter is an initializer
pub fn isInitializer(name: []const u8, initializers: []*TensorProto) bool {
    for (initializers) |init| {
        if (std.mem.eql(u8, init.name.?, name)) return true;
    }
    return false;
}

// Returns the corresponding TensorProto for the given name if it exists in the initializers list.
// Returns an error if the initializer is not found.
pub fn getInitializer(name: []const u8, initializers: []*TensorProto) !*TensorProto {
    for (initializers) |init| {
        if (std.mem.eql(u8, init.name.?, name)) return init;
    }

    return error.NotExistingInitializer;
}

// -------------------- PRINTERS --------------------

// Prints the list of nodes in the given computation graph.
// Outputs each node's name along with its input and output tensors and their readiness status.
pub fn printNodeList(graph: std.ArrayList(ReadyNode)) !void {
    for (graph.items) |node| {
        std.debug.print("\n ----- node: {s}", .{node.nodeProto.name.?});

        std.debug.print("\n          inputs: ", .{});
        // Write the inputs
        for (node.inputs.items) |input| {
            std.debug.print("\n              ->{s} {s}", .{ input.name, if (input.ready) "--->ready" else "" });
        }

        std.debug.print("\n          outputs:", .{});
        // Write the outputs
        for (node.outputs.items) |output| {
            std.debug.print("\n              -> {s} {s}", .{ output.name, if (output.ready) "--->ready" else "" });
        }
    }
}

// Prints the list of nodes that are ready for computation.
// Outputs each node's name, operation type, inputs, and outputs along with their readiness status.
pub fn printComputableNodes(computableNodes: std.ArrayList(*ReadyNode)) !void {
    for (computableNodes.items) |node| {
        std.debug.print("\n ----- node: {s}", .{node.nodeProto.name.?});
        std.debug.print("\n          op_type: {s}", .{node.nodeProto.op_type});
        std.debug.print("\n          inputs: {}", .{node.inputs.items.len});
        // Write the inputs
        for (node.inputs.items) |input| {
            std.debug.print("\n              -> {s} {s}", .{ input.name, if (input.ready) "--->ready" else return error.ShouldBeReady });
        }
        std.debug.print("\n          outputs:", .{});
        // Write the outputs
        for (node.outputs.items) |output| {
            std.debug.print("\n              -> {s} {s}", .{ output.name, if (output.ready) return error.OutputReadyTooEarly else "" });
        }
    }
}

// Prints the list of unique ONNX operations present in the given graph.
// Outputs each operation type only once.
pub fn printOperations(graph: *GraphProto) !void {
    std.debug.print("\n", .{});
    std.debug.print("\n-------------------------------------------------", .{});
    std.debug.print("\n+                ONNX operations                +", .{});
    std.debug.print("\n-------------------------------------------------", .{});

    var op_set = std.StringHashMap(void).init(std.heap.page_allocator);
    defer op_set.deinit();

    for (graph.nodes) |node| {
        try op_set.put(node.op_type, {});
    }

    var it = op_set.iterator();
    while (it.next()) |entry| {
        std.debug.print("\n- {s}", .{entry.key_ptr.*});
    }

    std.debug.print("\n-------------------------------------------------\n", .{});
}

// Function to print all entries in the tensorHashMap
pub fn printTensorHashMap(map: std.StringHashMap(ReadyTensor)) void {
    var it = map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const tensor = entry.value_ptr.*;
        std.debug.print("\nTensor Name: {s}", .{key});
        std.debug.print("\n     Ready: {}", .{tensor.ready});
        std.debug.print("\n     Shape: [{any}]", .{tensor.shape});
    }
}
