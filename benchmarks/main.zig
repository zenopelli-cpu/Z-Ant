const mat_mul_bench = @import("matmul/benchmark.zig");

pub fn main() !void {
    try mat_mul_bench.run();
}
