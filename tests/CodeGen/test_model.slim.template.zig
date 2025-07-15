const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen").codegen_v1;
const utils = codegen.utils;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const model = @import("model_options.zig");

const ITERATION_COUNT: u32 = 100;

test "Static Library - Random data Prediction Test" {
    std.testing.log_level = .info;

    std.debug.print("\ntest: Static Library - Model: {s}  - Random data Prediction Test -------------------------\n", .{model.name});

    var input_shape = model.input_shape;

    var input_data_size: u32 = 1;
    for (input_shape) |dim| {
        input_data_size *= dim;
    }

    // Create input data array directly instead of ArrayList
    var input_data = try allocator.alloc(model.input_data_type, input_data_size);
    defer allocator.free(input_data);

    // Generate random data
    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    for (0..ITERATION_COUNT) |_| {
        // Fill with random values
        for (0..input_data_size) |i| {
            input_data[i] = rand.float(model.input_data_type) * 100;
        }

        var result: [*]model.output_data_type = undefined;

        // Run prediction
        model.lib.predict(
            input_data.ptr,
            @ptrCast(&input_shape),
            input_shape.len,
            &result,
        );
    }
    std.debug.print("\nRan 100 fuzzy tests on model \"{s}\", done without errors:\n", .{model.name});
}

test "Static Library - Inputs Prediction Test" {
    std.testing.log_level = .info;

    std.debug.print("\ntest: Codegen one-op model: \"{s}\" compare with Pre-Generated results. -------------------------\n", .{model.name});

    var input_shape = model.input_shape;

    var input_data_len: u32 = 1;
    for (input_shape) |dim| {
        input_data_len *= dim;
    }

    // Read json for output to compare it with.
    const user_tests_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels/{s}/user_tests.json", .{model.name});
    defer allocator.free(user_tests_path);

    std.debug.print("{s}", .{user_tests_path});

    const parsed_user_tests = try utils.loadUserTests(model.input_data_type, model.output_data_type, user_tests_path);
    defer parsed_user_tests.deinit();

    const user_tests = parsed_user_tests.value;

    std.debug.print("\nUser tests loaded.\n", .{});

    for (user_tests) |user_test| {
        std.debug.print("\n\tRunning user test: {s}\n\n", .{user_test.name});

        try std.testing.expectEqual(user_test.input.len, input_data_len);

        var result: [*]model.output_data_type = undefined;

        // Run prediction
        model.lib.predict(
            user_test.input.ptr,
            @ptrCast(&input_shape),
            input_shape.len,
            &result,
        );

        for (0.., user_test.output) |i, expected_output| {
            const result_value = result[i];

            const big_diff: bool = expected_output - result_value > marginFor(model.output_data_type);
            if (big_diff)
                std.debug.print("\n\n  >>>>>>>ERROR!!<<<<<< \nTest failed for input: {d} expected: {} got: {}, margin: {}\n", .{ i, expected_output, result_value, marginFor(model.output_data_type) });
            try std.testing.expect(!big_diff);
        }

        std.debug.print("\n expected: {any} ", .{user_test.output});
        std.debug.print("\n obtained: {{", .{});
        for (0..user_test.output.len) |i| {
            if (i > 0) std.debug.print(",", .{});
            std.debug.print(" {}", .{result[i]});
        }
        std.debug.print(" }}", .{});
    }
}

/// Returns `true` if `T` is any integer type (signed or unsigned).
fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => true,
        else => false,
    };
}

/// Returns `0` (of type `T`) for integer `T`, otherwise `0.001` (of type `T`).
fn marginFor(comptime T: type) T {
    return if (@typeInfo(T) == .int)
        // integer types: zero tolerance
        0
    else
        // floating‑point (or any other type): tiny tolerance
        0.001;
}
