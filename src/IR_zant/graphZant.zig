const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;
const NodeZant = @import("nodeZant.zig").NodeZant;
const tensorZant_lib = @import("tensorZant.zig");

//--- proto structure
const GraphProto = zant.onnx.GraphProto;

// --- fusion
const pattern_matcher = @import("fusion/pattern_matcher.zig");
const PatternConfig = pattern_matcher.PatternConfig;

pub const GraphZant = struct {
    name: ?[]const u8,
    nodes: std.ArrayList(*NodeZant),
    graphProto: *GraphProto,

    /// Initializes a GraphZant instance.
    pub fn init(graphProto: *GraphProto) !GraphZant {
        return GraphZant{
            .name = graphProto.name.?,
            .nodes = std.ArrayList(*NodeZant).init(allocator),
            .graphProto = graphProto,
        };
    }

    /// Deinitializes a GraphZant instance, freeing allocated resources.
    pub fn deinit(self: *GraphZant) void {
        // Deinitialize each NodeZant in the nodes list
        for (self.nodes.items) |node| {
            node.deinit();
            allocator.destroy(node); // Free the node
        }
        self.nodes.deinit();
    }

    // Adds a new node to the graph.
    pub fn build_graph(self: *GraphZant) !void {

        // create all the nodes
        for (self.graphProto.nodes) |nodeProto| {
            // allocate memory for the node
            const node = try allocator.create(NodeZant);
            node.* = try NodeZant.init(nodeProto);
            try self.nodes.append(node);
        }

        //hashmap for the outputs for the producers
        var output_map = std.StringHashMap(*NodeZant).init(allocator);
        defer output_map.deinit();

        //populate the output map with the nodes
        for (self.nodes.items) |node| {
            for (node.nodeProto.?.output) |output| {
                try output_map.put(output, node);
            }
        }

        // use the hashmap to find the producers of the inputs
        for (self.nodes.items) |customer| {
            for (customer.nodeProto.?.input) |input| {
                const producer = output_map.get(input);
                if (producer) |p| {
                    try p.add_next(customer);
                }
            }
        }

        // Perform shape inference after graph is built
        //try self.performShapeInference(&output_map);
    }

    // /// Mathematical shape inference pass over the entire graph
    // fn performShapeInference(self: *GraphZant, output_map: *std.StringHashMap(*NodeZant)) !void {
    //     std.debug.print("\n=== STARTING MATHEMATICAL SHAPE INFERENCE ===\n", .{});

    //     // Multiple passes to propagate shapes through the graph
    //     const max_passes: u32 = 5;
    //     var pass: u32 = 0;

    //     while (pass < max_passes) {
    //         std.debug.print("Shape inference pass {}\n", .{pass + 1});
    //         var shapes_updated = false;

    //         // Go through nodes in dependency order and compute output shapes
    //         for (self.nodes.items) |node| {
    //             const node_updated = try self.computeNodeOutputShapes(node, output_map);
    //             if (node_updated) {
    //                 shapes_updated = true;
    //                 std.debug.print("Updated shapes for node: {s}\n", .{node.op_type});
    //             }
    //         }

    //         // If no shapes were updated, we're done
    //         if (!shapes_updated) {
    //             std.debug.print("Shape inference converged after {} passes\n", .{pass + 1});
    //             break;
    //         }

    //         pass += 1;
    //     }

    //     std.debug.print("=== SHAPE INFERENCE COMPLETE ===\n\n", .{});
    // }

    // /// Compute output shapes for a single node based on its operation type
    // fn computeNodeOutputShapes(self: *GraphZant, node: *NodeZant, output_map: *std.StringHashMap(*NodeZant)) !bool {
    //     _ = self;
    //     _ = output_map;

    //     // Try to compute the correct shape first
    //     const new_shape = node.op.get_output_shape() catch {
    //         return false;
    //     };

    //     // Get output tensors for this node
    //     const output_tensors = node.get_output_tensors() catch return false;
    //     // Note: potential memory leak, but avoids segfault for now

    //     var shapes_changed = false;

    //     for (output_tensors) |output_tensor| {
    //         // Only update LINK tensors with placeholder shapes
    //         if (output_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
    //             // Check if current shape is placeholder
    //             const is_placeholder = (output_tensor.shape.len == 1 and output_tensor.shape[0] == 1);

    //             if (is_placeholder) {
    //                 // Update the tensor shape with the computed shape
    //                 // NOTE: Potential memory leak, but avoids segfault
    //                 // TODO: Implement proper shape memory tracking
    //                 output_tensor.shape = try allocator.dupe(usize, new_shape);
    //                 output_tensor.stride = try tensorZant_lib.TensorZant.computeStride(output_tensor.shape);

    //                 std.debug.print("Shape inference: Updated {s} from placeholder to {any}\n", .{ output_tensor.name, output_tensor.shape });
    //                 shapes_changed = true;
    //             }
    //         }
    //     }

    //     return shapes_changed;
    // }

    // kernel fusion and sobstitutions
    pub fn fuse(self: *GraphZant, pattern_configs: []const PatternConfig) !void {
        std.debug.print("\nGraphZant.fuse()...", .{});
        try pattern_matcher.fusePatterns(self, pattern_configs);
    }

    // linearize the graph
    pub fn linearize(self: *GraphZant, alloc: std.mem.Allocator) !std.ArrayList(*NodeZant) {
        var visited = std.AutoHashMap(*NodeZant, bool).init(alloc);
        var result = std.ArrayList(*NodeZant).init(alloc);
        defer visited.deinit();

        for (self.nodes.items) |node| {
            try dfs(node, &visited, &result);
        }

        GraphZant.reverseArrayList(*NodeZant, &result);

        return result;
    }

    pub fn dfs(
        node: *NodeZant,
        visited: *std.AutoHashMap(*NodeZant, bool),
        result: *std.ArrayList(*NodeZant),
    ) !void {
        if (visited.get(node)) |_| return;

        try visited.put(node, true);

        for (node.next.items) |child| {
            try dfs(child, visited, result);
        }

        try result.append(node);
    }

    pub fn reverseArrayList(comptime T: type, list: *std.ArrayList(T)) void {
        var i: usize = 0;
        var j: usize = list.items.len - 1;
        while (i < j) {
            const temp = list.items[i];
            list.items[i] = list.items[j];
            list.items[j] = temp;

            i += 1;
            j -= 1;
        }
    }

    pub fn print_before_linearizzation(self: *GraphZant) void {
        std.debug.print("\n\nGraphZant: {s}\n", .{self.name orelse "<unnamed>"});
        for (self.nodes.items) |node| {
            node.print();
        }
    }

    // Splits a linearized graph of NodeZant into two partitions:
    // one that fits within `max_edge_memory` (bytes), and one for server execution.
    pub fn splitNodesByMemory(nodes: std.ArrayList(*NodeZant), max_edge_memory: usize) !struct {
        edge_nodes: std.ArrayList(*NodeZant),
        server_nodes: std.ArrayList(*NodeZant),
    } {
        var edge_nodes = std.ArrayList(*NodeZant).init(allocator);
        var server_nodes = std.ArrayList(*NodeZant).init(allocator);

        var cumulative_memory: usize = 0;

        var i: usize = 0;
        while (i < nodes.items.len) : (i += 1) {
            const node = nodes.items[i];
            const node_mem = try node.op.get_memory_footprint();

            if (cumulative_memory + node_mem <= max_edge_memory) {
                try edge_nodes.append(node);
                cumulative_memory += node_mem;
            } else {
                break;
            }
        }

        // append the remaining nodes to the server
        while (i < nodes.items.len) : (i += 1) {
            try server_nodes.append(nodes.items[i]);
        }

        return .{
            .edge_nodes = edge_nodes,
            .server_nodes = server_nodes,
        };
    }
};
