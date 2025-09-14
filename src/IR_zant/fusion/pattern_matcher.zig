//! The aim of this is to detect well-defined sequences of nodes and sobsitute them with the fused version of those.
//! For Example:
//! Imagine in your graph you have the following sequence
//!     ...
//!     DeQuantLinear
//!     Pad
//!     QuantLinear
//!     Conv
//!     ...
//! it can be sobsitute with:
//!     ...
//!     qlinarConv
//!     ...
//! qlinarConv is a known operation the ONNX format, but for example if I want I can fuse the following:
//!     ...
//!     Conv
//!     Relu
//!     ...
//! into a custom operation called fused_Conv_Relu.
//!
//! RULES:
//!     - The name of the operations must be a concatenation of the string "fused" with names of the operations in the same order of the fusion, so
//!     the sequence: opA -> opB -> opC will be fused, if possible, into "fused_opA_opB_opC"
//!     - The pattern matcher will only fuse a math operation into an already existing math kernel
//!     - The names of the input and output tensors will be the same of the input and output tensors of the fused sequence
//!
//! Pattern Validation Logic:
//!
//!     - Start Node: Can have any number of incoming connections
//!     - Middle Nodes: Must have exactly 1 outgoing connection (no branching allowed)
//!     - End Node: Can have multiple outgoing connections (preserved in fusion)
//!     - This ensures that when we fuse DequantizeLinear -> Pad -> QuantizeLinear -> QLinearConv:
//!             -Each intermediate step has a clear, unambiguous next node
//!             - The final QLinearConv node can connect to multiple subsequent operations
//!             - No graph information is lost during the fusion process
//!     - The pattern matching will fail (return null) if any middle node has multiple outgoing connections, preventing incorrect fusions that would lose graph topology information.

const std = @import("std");
const allocator = std.heap.page_allocator; // Use your allocator

// --- Zant_IR ---
const IR_zant = @import("../IR_zant.zig");
const GraphZant = IR_zant.GraphZant;
const NodeZant = IR_zant.NodeZant;

const fuse_dequant_pad_qant_qLinConv = @import("fusion_operations.zig").fuse_dequant_pad_qant_qLinConv;

// Configuration for a fusion pattern
pub const PatternConfig = struct {
    pattern: []const []const u8, // Array of operation type strings
    fusion_fn: *const fn (std.ArrayList(*NodeZant)) anyerror!NodeZant,
    name: []const u8,
    pattern_fn: ?*const fn (graph: *GraphZant, root_node: *NodeZant) anyerror!NodeZant, //used for more complex pattern like detect_qadd_pattern()
};

/// Generic pattern matcher that can be extended for other patterns
pub fn fusePatterns(graph: *GraphZant, pattern_configs: []const PatternConfig) !void {
    for (pattern_configs) |config| {
        try fusePatternsByConfig(graph, config);
    }
}

/// Generic pattern matcher using configuration
fn fusePatternsByConfig(graph: *GraphZant, config: PatternConfig) !void {
    var i: usize = 0;

    while (i < graph.nodes.items.len) {
        const node = graph.nodes.items[i];

        // Only start pattern matching from nodes that match the first operation in pattern
        if (std.mem.eql(u8, node.op_type, config.pattern[0])) {
            if (try findAndFusePattern(graph, node, config)) {
                // Pattern was found and fused, restart search since graph was modified
                i = 0;
                continue;
            }
        }
        i += 1;
    }
}

/// Generic pattern finder using configuration - follows actual graph connections
fn findAndFusePattern(graph: *GraphZant, start_node: *NodeZant, config: PatternConfig) !bool {
    // Try to find the complete pattern starting from this node
    if (if (config.pattern_fn) |custom_pattern_search| try custom_pattern_search(graph, start_node) else try findPatternFromNode(start_node, config.pattern)) |pattern_nodes| {
        defer pattern_nodes.deinit();

        // Create fused node
        const fused_node = try allocator.create(NodeZant);
        fused_node.* = try config.fusion_fn(pattern_nodes);

        // Update graph structure
        try replacePatternWithFusedNode(graph, pattern_nodes, fused_node);

        return true;
    }

    return false;
}

/// Find a complete pattern by following node connections
fn findPatternFromNode(start_node: *NodeZant, pattern: []const []const u8) !?std.ArrayList(*NodeZant) {
    if (pattern.len == 0) return null;

    // Check if start node matches first pattern element
    if (!std.mem.eql(u8, start_node.op_type, pattern[0])) {
        return null;
    }

    var pattern_nodes = std.ArrayList(*NodeZant).init(allocator);
    try pattern_nodes.append(start_node);

    var current_node = start_node;

    // Follow connections to match the rest of the pattern
    for (pattern[1..], 1..) |expected_op, pattern_index| {
        // For nodes in the middle of pattern (not the last), they should have exactly one next connection
        // Only the last node in the pattern can have multiple outgoing connections
        const is_last_in_pattern = (pattern_index == pattern.len - 1);

        if (!is_last_in_pattern and current_node.next.items.len != 1) {
            // Middle nodes should have exactly one connection to avoid ambiguity
            pattern_nodes.deinit();
            return null;
        }

        var found_next = false;

        // Look through next connections to find matching operation
        for (current_node.next.items) |next_node| {
            if (std.mem.eql(u8, next_node.op_type, expected_op)) {
                try pattern_nodes.append(next_node);
                current_node = next_node;
                found_next = true;
                break;
            }
        }

        // If we didn't find the expected next operation, pattern doesn't match
        if (!found_next) {
            pattern_nodes.deinit();
            return null;
        }
    }

    return pattern_nodes;
}

/// Replace a pattern of nodes with a single fused node in the graph
fn replacePatternWithFusedNode(graph: *GraphZant, pattern_nodes: std.ArrayList(*NodeZant), fused_node: *NodeZant) !void {
    const first_node = pattern_nodes.items[0];
    const last_node = pattern_nodes.items[pattern_nodes.items.len - 1];

    // Redirect incoming connections to first node -> fused node
    try redirectIncomingConnections(graph, first_node, fused_node);

    // Set fused node's next connections to match last node's connections
    fused_node.next = last_node.next;

    // Remove all pattern nodes from the graph
    for (pattern_nodes.items) |pattern_node| {
        for (graph.nodes.items, 0..) |graph_node, i| {
            if (graph_node == pattern_node) {
                _ = graph.nodes.orderedRemove(i);
                break;
            }
        }
    }

    // Add the fused node to the graph
    try graph.nodes.append(fused_node);
}

/// Redirect all nodes that point to old_node to point to new_node instead
fn redirectIncomingConnections(graph: *GraphZant, old_node: *NodeZant, new_node: *NodeZant) !void {
    for (graph.nodes.items) |node| {
        for (node.next.items, 0..) |next_node, j| {
            if (next_node == old_node) {
                node.next.items[j] = new_node;
            }
        }
    }
}
