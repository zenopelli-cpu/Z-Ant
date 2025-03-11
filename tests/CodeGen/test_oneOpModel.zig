const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const oneOp = @import("oneOpModelGenerator.zig");

test "OneOptest Add" {
    std.debug.print("\n     test: OneOptest Add", .{});

    var shape = [_]i64{ 1, 2, 3, 4 };
    var myStruct = oneOp.AddStruct{
        .name = "Add",
        .inputShape = shape[0..],
    };
    _ = &myStruct;

    var myModel = try oneOp.oneOpModel(oneOp.OpStruct{ .Add = myStruct });
    defer myModel.deinit(allocator);

    myModel.print();
}
