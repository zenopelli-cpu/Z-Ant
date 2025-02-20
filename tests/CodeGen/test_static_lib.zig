const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkgAllocator = @import("pkgAllocator");
const allocator = pkgAllocator.allocator;

test "Static Library - Basic Prediction Test" {
    std.debug.print("\n     test: Static Library - Basic Prediction Test\n", .{});

    // Create a simple input tensor
    var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var input_shape = [_]u32{5};
    var result: [*]f32 = undefined;

    // Create a logging function
    const LogFn = fn ([*c]u8) callconv(.C) void;
    const logFn: LogFn = struct {
        fn log(msg: [*c]u8) callconv(.C) void {
            std.debug.print("{s}", .{msg});
        }
    }.log;

    // Set the logging function
    @import("static_lib").setLogFunction(logFn);

    // Run prediction
    @import("static_lib").predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape),
        1,
        &result,
    );

    // Print the results
    std.debug.print("\nPrediction results:\n", .{});
    for (0..5) |i| {
        std.debug.print("Result[{d}] = {d:.6}\n", .{ i, result[i] });
    }

    // Verify the results sum to approximately 1.0 (since we use softmax at the end)
    var sum: f32 = 0;
    for (0..5) |i| {
        sum += result[i];
    }
    const epsilon = 0.0001;
    try std.testing.expect(sum >= 1.0 - epsilon and sum <= 1.0 + epsilon);
}

test "Static Library - Error Cases" {
    std.debug.print("\n     test: Static Library - Error Cases\n", .{});

    // Test with empty shape
    {
        var input_data = [_]f32{};
        var input_shape = [_]u32{};
        var result: [*]f32 = undefined;

        @import("static_lib").predict(
            @ptrCast(&input_data),
            @ptrCast(&input_shape),
            0,
            &result,
        );
    }

    // Test with invalid shape length
    {
        var input_data = [_]f32{ 1.0, 2.0 };
        var input_shape = [_]u32{ 1, 2 }; // Wrong shape for our network
        var result: [*]f32 = undefined;

        @import("static_lib").predict(
            @ptrCast(&input_data),
            @ptrCast(&input_shape),
            2,
            &result,
        );
    }
}
