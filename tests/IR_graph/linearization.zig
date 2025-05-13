const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const NodeZant = zant.IR_graph.NodeZant;
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.tensor.Tensor;

test "linearizing mnist-8 " {
    std.debug.print("\n\n ------TEST: linearizing mnist-8 ", .{});

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: zant.IR_graph.GraphZant = try zant.IR_graph.init(&model);
    defer graphZant.deinit();

    const linearizedGraph = try graphZant.linearize(allocator);

    std.debug.print("\n\nLinearized Graph Nodes:\n", .{});
    for (linearizedGraph.items) |node| {
        std.debug.print(" - Node: {s}\n", .{node.nodeProto.name orelse "<unnamed>"});
    }

    linearizedGraph.deinit();
}
