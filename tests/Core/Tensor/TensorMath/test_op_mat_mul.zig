const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const ErrorHandler = zant.utils.error_handler;

test "new MatMul Square N" {

    const allocator = pkgAllocator.allocator;
    const N = std.math.pow(usize, 2, 7);
    
    std.debug.print("\n     test:MatMul Square {d}\n", .{N});
    var shape: [2]usize = [_]usize{ N, N }; // 2x2 matrix

    // Generate two random array with 2^5 x 2^5

    var input_data_size: usize = 1;
    for (shape) |dim| {
        input_data_size *= dim;
    }

    // Create input data array directly instead of ArrayList
    var inputArray1 = try allocator.alloc(f32, input_data_size);
    defer allocator.free(inputArray1);
    var inputArray2 = try allocator.alloc(f32, input_data_size);
    defer allocator.free(inputArray2);

    // Generate random data
    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    // Fill with random values
    for (0..input_data_size) |i| {
        inputArray1[i] = rand.float(f32) * 100;
        inputArray2[i] = rand.float(f32) * 100;
    }
    
    //Allocate input tensors
    var t1 = try Tensor(f32).fromArray(&allocator, inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, inputArray2, &shape);
    defer t2.deinit();
    
    //Allocate output tensors
    // Calculate time 
    
    // Benchmark old implementation
    var timer = try std.time.Timer.start();
    const old_start = timer.lap();
    var old = try TensMath.mat_mul(f32, &t1, &t2);
    defer old.deinit();
    const old_end = timer.lap();
    const old_time_ns = old_end - old_start;
    
    // Benchmark new implementation
    const new_start = timer.lap();
    var new = try TensMath.new_mat_mul(f32, &t1, &t2);
    const new_end = timer.lap();
    defer new.deinit();
    const new_time_ns = new_end - new_start;
    
    // Print benchmark results
    std.debug.print("\nBenchmark Results:\n", .{});
    std.debug.print("Original mat_mul: {d} ns ({d:.3} ms)\n", .{old_time_ns, @as(f64, @floatFromInt(old_time_ns)) / 1_000_000.0});
    std.debug.print("New mat_mul:     {d} ns ({d:.3} ms)\n", .{new_time_ns, @as(f64, @floatFromInt(new_time_ns)) / 1_000_000.0});
    
    // Calculate speedup
    const speedup = @as(f64, @floatFromInt(old_time_ns)) / @as(f64, @floatFromInt(new_time_ns));
    std.debug.print("Speedup:         {d:.2}x\n\n", .{speedup});
    
    // std.debug.print("Printing OLD: \n", .{});
    // for (0..N) |x| {
    //     for (0..N) |y| {
    //         std.debug.print("{d:3} ", .{old.data[x*N+y]});
    //     }
    //     std.debug.print("\n", .{});
    // }
    
    // std.debug.print("Printing NEW: \n", .{});
    // for (0..N) |x| {
    //     for (0..N) |y| {
    //         std.debug.print("{d:3} ", .{new.data[x*N+y]});
    //     }
    //     std.debug.print("\n", .{});
    // }

    for (0..N) |x| {
        for (0..N) |y| {
            try std.testing.expect(old.data[x*N+y] == new.data[x*N+y]);
        }
    }
    
    
}

// test "MatMul 2x2" {
//     std.debug.print("\n     test:MatMul 2x2", .{});

//     const allocator = pkgAllocator.allocator;

//     var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

//     var inputArray: [2][2]f32 = [_][2]f32{
//         [_]f32{ 1.0, 2.0 },
//         [_]f32{ 4.0, 5.0 },
//     };

//     var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
//     var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);

//     var result_tensor = try TensMath.mat_mul(f32, &t1, &t2);

//     try std.testing.expect(9.0 == result_tensor.data[0]);
//     try std.testing.expect(12.0 == result_tensor.data[1]);

//     result_tensor.deinit();
//     t1.deinit();
//     t2.deinit();
// }

test "Error when input tensors have incompatible sizes for MatMul" {
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.mat_mul(f32, &t1, &t2));

    _ = TensMath.mat_mul(f32, &t1, &t2) catch |err| {
        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible shapes for MatMul" {
    std.debug.print("\n     test: Error when input tensors have incompatible shapes for MatMul", .{});
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 4x1 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.mat_mul(f32, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "Compare MatMul implementations with execution time" {
    std.debug.print("\nTest: Compare MatMul implementations with execution time\n", .{});
    const allocator = pkgAllocator.allocator;

    // Create test tensors
    var shape: [2]usize = [_]usize{ 16, 16 };
    var matrix1: [16][16]f32 = undefined;
    var matrix2: [16][16]f32 = undefined;

    // Initialize with different values for each matrix
    for (0..16) |i| {
        for (0..16) |j| {
            matrix1[i][j] = @floatFromInt(i * 16 + j + 1);
            matrix2[i][j] = @floatFromInt((15 - i) * 16 + (15 - j) + 1); // Reverse pattern
        }
    }

    var t1 = try Tensor(f32).fromArray(&allocator, &matrix1, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &matrix2, &shape);
    defer t1.deinit();
    defer t2.deinit();

    // Run multiple iterations for more accurate timing
    const iterations = 100;
    var total_simd: i64 = 0;
    var total_flat: i64 = 0;

    for (0..iterations) |_| {
        const start_simd = std.time.nanoTimestamp();
        var result_simd = try TensMath.mat_mul(f32, &t1, &t2);
        const end_simd = std.time.nanoTimestamp();
        total_simd += @as(i64, @intCast(end_simd - start_simd));
        result_simd.deinit();

        const start_flat = std.time.nanoTimestamp();
        var result_flat = try TensMath.dot_product_tensor_flat(f32, f32, &t1, &t2);
        defer result_flat.deinit();
        const end_flat = std.time.nanoTimestamp();
        total_flat += @as(i64, @intCast(end_flat - start_flat));
        result_flat.deinit();
    }

    const avg_simd = @divFloor(total_simd, iterations);
    const avg_flat = @divFloor(total_flat, iterations);

    std.debug.print("Average over {d} iterations:\n", .{iterations});
    std.debug.print("SIMD execution time: {d} ns\n", .{avg_simd});
    std.debug.print("Flat execution time: {d} ns\n", .{avg_flat});
    std.debug.print("SIMD is {d:.2}x {s}\n", .{ if (avg_simd < avg_flat)
        @as(f64, @floatFromInt(avg_flat)) / @as(f64, @floatFromInt(avg_simd))
    else
        @as(f64, @floatFromInt(avg_simd)) / @as(f64, @floatFromInt(avg_flat)), if (avg_simd < avg_flat) "faster" else "slower" });
}
