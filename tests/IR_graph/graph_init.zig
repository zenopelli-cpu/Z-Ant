const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.tensor.Tensor;

test "parsing mnist-8 graphZant" {
    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, "datasets/models/mnist-8/mnist-8.onnx");
    defer model.deinit(allocator);

    //model.print();

    var graphZant: zant.IR_graph.GraphZant = try zant.IR_graph.init(&model);
    defer graphZant.deinit();

    //USELESS SHIT FOR DEBUG
    // std.debug.print("__HASH_MAP__", .{});
    // var it = zant.IR_graph.tensorZant_lib.tensorMap.iterator();
    // while (it.next()) |entry| {
    //     std.debug.print("Key: {s}, Value.ty: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.name });
    // }
}
