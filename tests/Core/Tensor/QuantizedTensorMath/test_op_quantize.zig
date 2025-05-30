const std = @import("std");
const zant = @import("zant");

const QuantMath = zant.core.tensor.quantized_math;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator.allocator;

const testing = std.testing;

// asymmetric quantization

test "asymm signed quantization" {
    std.debug.print("\n    test: asymm signed quantization\n", .{});

    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.234, 0.3435, -0.231 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var inputTensor = try Tensor(f32).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();
    inputTensor.printMultidim();

    var outputTensor = try QuantMath.quantize(f32, i8, &inputTensor, QuantMath.quantScheme.ASYM);
    defer outputTensor.deinit();
    outputTensor.printMultidim();

    try testing.expectEqual(5.098039e-3, try outputTensor.get_scale_factor());
    try testing.expectEqual(98, try outputTensor.get_zero_point());
}
