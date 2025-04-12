const zant = @import("zant");
const bench_options = @import("bench_options");
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;
const TensMath = zant.core.tensor.math_standard;
const std = @import("std");

const macs_mat_mul = @import("matmul/macs_matmul.zig").lean_mat_mul;
const simple_mat_mul = @import("matmul/simple_matmul.zig").lean_mat_mul;
const blocked_matmul = @import("matmul/blocked_matmul.zig").lean_mat_mul;

const mat_mul_fn = fn (comptime T: anytype, A: anytype, B: anytype, C: anytype) callconv(.@"inline") anyerror!void;

const Benchmark = struct {
    name: []const u8,
    mat_mul: mat_mul_fn,
};

const test_bench = [_]Benchmark{
    .{ .name = "CurrentMatMul", .mat_mul = TensMath.lean_matmul },
    .{ .name = "BlockedMatMul", .mat_mul = blocked_matmul },
};

const test_data_types = [_]type{
    u8,
    u16,
    u32,
    f16,
    f32,
};

const test_data_n = [_]usize{ 8, 9, 10 };

fn print_mat(comptime T: anytype, mat: *const Tensor(T)) void {
    std.debug.print("Matrix shape: ", .{});
    for (mat.shape) |dim| std.debug.print("{} ", .{dim});
    std.debug.print("\n", .{});

    for (0..mat.shape[0]) |i| {
        for (0..mat.shape[1]) |j| {
            std.debug.print("{d} ", .{mat.data[i * mat.shape[1] + j]});
        }
        std.debug.print("\n", .{});
    }
}

inline fn mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), lean_mat_mul: mat_mul_fn) !Tensor(T) {
    // std.debug.print("\nStarting matrix multiplication validation...\n", .{});

    const dim_num = A.shape.len;

    // Create output tensor

    var out_shape = try allocator.alloc(usize, dim_num);
    defer allocator.free(out_shape);
    errdefer allocator.free(out_shape);

    // Copy all dimensions except the last two
    for (0..(dim_num - 2)) |i| {
        out_shape[i] = A.shape[i];
    }

    // Set the last two dimensions to the dimensions of the input tensors
    out_shape[dim_num - 2] = A.shape[dim_num - 2];
    out_shape[dim_num - 1] = B.shape[dim_num - 1];

    // Create output tensor

    var Y = try Tensor(T).fromShape(&allocator, out_shape);
    errdefer Y.deinit();

    // std.debug.print("Output tensor shape: ", .{});
    // for (Y.shape) |dim| std.debug.print("{} ", .{dim});
    // std.debug.print("\n", .{});

    @memset(Y.data, 0);
    
    try lean_mat_mul(T, A, B, &Y);

    return Y;
}

fn run_mat_mul_benchmark(comptime T: anytype, comptime N: u8, comptime square_only: bool) !void {
    std.debug.print("Running square matrix multiplication benchmark for N = {d}, type = {any}\n", .{ N, T });

    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const twoN = std.math.pow(usize, 2, N);
    const data_type = T;

    const cache_block_size = std.atomic.cache_line / @sizeOf(data_type);

    var rnd_a_rows: usize = 0;
    var rnd_a_cols: usize = 0;
    var rnd_b_cols: usize = 0;

    var shape_1: [2]usize = [_]usize{ twoN, twoN }; // 2^N x 2^N matrix
    var shape_2: [2]usize = [_]usize{ twoN , twoN }; // 2^N x 2^N matrix

    if (!square_only) {
        rnd_a_rows = rand.intRangeLessThan(u8, 1, cache_block_size);
        rnd_a_cols = rand.intRangeLessThan(u8, 1, cache_block_size);
        rnd_b_cols = rand.intRangeLessThan(u8, 1, cache_block_size);

        shape_1[0] += rnd_a_rows;
        shape_1[1] += rnd_a_cols;
        shape_2[0] += rnd_a_cols;
        shape_2[1] += rnd_b_cols;
    }

    // Print the shapes
    std.debug.print("Shape 1: ", .{});
    for (shape_1) |dim| std.debug.print("{d} ", .{dim});
    std.debug.print("\n", .{});

    std.debug.print("Shape 2: ", .{});
    for (shape_2) |dim| std.debug.print("{d} ", .{dim});
    std.debug.print("\n", .{});

    var input_data_size: usize = 1;
    for (shape_1) |dim| {
        input_data_size *= dim;
    }

    // Create input data array directly instead of ArrayList
    var inputArray1 = try allocator.alloc(data_type, input_data_size);
    defer allocator.free(inputArray1);

    // Fill with random values
    for (0..input_data_size) |i| {
        if (data_type == f32 or data_type == f64 or data_type == f16) {
            inputArray1[i] = @floatFromInt(rand.intRangeLessThan(u4, 0, 10));
        } else {
            inputArray1[i] = rand.intRangeLessThan(u4, 0, 10);
        }
    }

    input_data_size = 1;
    for (shape_2) |dim| {
        input_data_size *= dim;
    }

    var inputArray2 = try allocator.alloc(data_type, input_data_size);
    defer allocator.free(inputArray2);

    // Fill with random values
    for (0..input_data_size) |i| {
        if (data_type == f32 or data_type == f64 or data_type == f16) {
            inputArray2[i] = @floatFromInt(rand.intRangeLessThan(u4, 0, 10));
        } else {
            inputArray2[i] = rand.intRangeLessThan(u4, 0, 10);
        }
    }

    //Allocate input tensors
    var t1 = try Tensor(data_type).fromArray(&allocator, inputArray1, &shape_1);
    defer t1.deinit();
    var t2 = try Tensor(data_type).fromArray(&allocator, inputArray2, &shape_2);
    defer t2.deinit();

    //Allocate output tensors
    // Calculate time

    // Benchmark old implementation
    var timer = try std.time.Timer.start();
    const simple_start = timer.lap();
    var simple = try mat_mul(data_type, &t1, &t2, simple_mat_mul);
    defer simple.deinit();
    const simple_end = timer.lap();
    const simple_time_ns = simple_end - simple_start;

    var bench_times: [test_bench.len]u64 = undefined;

    // Running the benchmarks
    inline for (0..test_bench.len, test_bench) |i, bench| {
        std.debug.print("Running: {s}\n", .{bench.name});
        timer.reset();

        const start = timer.lap();
        var result = try mat_mul(data_type, &t1, &t2, bench.mat_mul);

        defer result.deinit();

        const end = timer.lap();
        const time_ns = end - start;

        bench_times[i] = time_ns;

        // Check if the result is correct

        check_loop: for (0..simple.shape[0]) |x| {
            for (0..simple.shape[1]) |y| {
                if (simple.data[x * simple.shape[1] + y] != result.data[x * simple.shape[1] + y]) {
                    std.debug.panic("Error: Element ({d}, {d}) mismatch: \n \t expected: {d} got {d}\n", .{ x, y, simple.data[x * simple.shape[1] + y], result.data[x * simple.shape[1] + y] });

                    break :check_loop;
                }
            }
        }
    }

    std.debug.print("Benchmark reuslts:\n", .{});

    std.debug.print("\tTimes:\n", .{});

    std.debug.print("\t\t{s} took {d} ms\n", .{ "Simple", @as(f64, @floatFromInt(simple_time_ns)) / 1_000_000.0 });

    inline for (0..test_bench.len, test_bench) |i, bench| {
        std.debug.print("\t\t{s} took {d} ms\n", .{ bench.name, @as(f64, @floatFromInt(bench_times[i])) / 1_000_000.0 });
    }

    std.debug.print("\tSpeedups:\n", .{});

    inline for (0..test_bench.len, test_bench) |i, bench| {
        const simple_speedup = @as(f64, @floatFromInt(simple_time_ns)) / @as(f64, @floatFromInt(bench_times[i]));
        const current_speedup = @as(f64, @floatFromInt(bench_times[0])) / @as(f64, @floatFromInt(bench_times[i]));
        std.debug.print("\t\t{s}:\n", .{bench.name});
        std.debug.print("\t\t\tSimple Speedup: {d:.2}x\n", .{simple_speedup});
        std.debug.print("\t\t\tCurrent Speedup: {d:.2}x\n", .{current_speedup});
        std.debug.print("\n", .{});
    }
}

pub fn main() !void {
    if (bench_options.full) {
        inline for (0..test_data_types.len) |i| {
            const T = test_data_types[i];
            inline for (0..test_data_n.len) |j| {
                const N = test_data_n[j];
                try run_mat_mul_benchmark(T, N, false);
                std.debug.print("\n\n", .{});
            }
        }
    } else {
        const N = 10; // Change this value to test different sizes
        const data_type = u8;
        try run_mat_mul_benchmark(data_type, N, false);
    }
}
