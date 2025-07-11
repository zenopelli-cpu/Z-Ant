const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

const NodeZant = IR_zant.NodeZant;
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.tensor.Tensor;

const ErrorDetail = struct {
    modelName: []const u8,
    errorLoad: anyerror,
};

var models: std.ArrayList([]const u8) = std.ArrayList([]const u8).init(allocator);
var failed_parsed_models: std.ArrayList(ErrorDetail) = std.ArrayList(ErrorDetail).init(allocator);

test "linearizing mnist-8 " {
    std.debug.print("\n\n ------TEST: linearizing mnist-8 ", .{});

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: IR_zant.GraphZant = try IR_zant.init(&model);
    defer graphZant.deinit();

    const linearizedGraph = try graphZant.linearize(allocator);
    defer linearizedGraph.deinit();

    std.debug.print("\n\nLinearized Graph Nodes:\n", .{});
    for (linearizedGraph.items) |node| {
        std.debug.print(" - Node: {s}\n", .{node.nodeProto.name orelse "<unnamed>"});
    }
}
