const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const utils = codegen.utils;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const model = @import("model_options.zig");

test "Static Library - Random data Prediction Test" {
    std.debug.print("\n     test: Static Library - Model: {s}  - Random data Prediction Test\n", .{model.name});

    var input_shape = model.input_shape;

    var input_data_size: u32 = 1;
    for (input_shape) |dim| {
        input_data_size *= dim;
    }

    // Create input data array directly instead of ArrayList
    var input_data = try allocator.alloc(model.data_type, input_data_size);
    defer allocator.free(input_data);

    // Generate random data
    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    // Fill with random values
    for (0..input_data_size) |i| {
        input_data[i] = rand.float(model.data_type) * 100;
    }

    var result: [*]model.data_type = undefined;

    // Run prediction
    model.lib.predict(
        input_data.ptr,
        @ptrCast(&input_shape),
        input_shape.len,
        &result,
    );

    std.debug.print("\nPrediction done without errors:\n", .{});
}
