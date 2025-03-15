const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const oneOp = @import("oneOpModelGenerator.zig");

test "OneOptests" {
    std.debug.print("\n     test: OneOptests", .{});

    try oneOp.oneOpModelsCodegen();
}
