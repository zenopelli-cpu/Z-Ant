const std = @import("std");
const zant = @import("zant");

const TensorWrapper = zant.core.tensorWrapper.TensorWrapper;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const pkgAllocator = zant.utils.allocator.allocator;

const expect = std.testing.expect;

test "Tensor test description" {
    std.debug.print("\n--- Running TensorWrapper tests\n", .{});
}

test "TensorWrapper CreateWrapper test" {
    // classic tensor
    var tensor = try Tensor(f64).init(&pkgAllocator);
    defer tensor.deinit();

    // wrap the classic tensor
    // var tensorWrapper = try TensorWrapper(@TypeOf(tensor), f64).createWrapper(&tensor);
    var tensorWrapper = try tensor.initWrapper();
    defer tensorWrapper.deinit();

    // test
    expect(tensorWrapper != null);
}
