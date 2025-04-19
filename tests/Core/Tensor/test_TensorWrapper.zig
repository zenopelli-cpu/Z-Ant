const std = @import("std");
const zant = @import("zant");

const TW = @import("../../../src/Core/Tensor/TensorWrapper.zig"); // TODO fix import
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
    // version 1
    //var tensorWrapper = try TW.TensorWrapper(@TypeOf(tensor), f64).createWrapper(&tensor);
    // version 2
    //var tensorWrapper = try tensor.initWrapper();
    // version 3
    const WrapperType = TW.TensorWrapper(@TypeOf(tensor), f64);
    var tensorWrapper = try WrapperType.createWrapper(&tensor);
    
    var tensorCopy = tensorWrapper.copy();
    defer tensorCopy.deinit();
    tensorCopy.printMultidim();

    defer tensorWrapper.deinit();

    // test wrapped tensor size
    expect(tensor.getSize() == tensorWrapper.getSize());
}

// pub fn initWrapper() TensorWrapper(@TypeOf(@This()), f64) {
//     return TensorWrapper(@TypeOf(@This()), f64).createWrapper(&@This());
// }