const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const assert = std.debug.assert;

const ArchitectureError = @import("errorHandler").ArchitectureError;
const TensorMathError = @import("errorHandler").TensorMathError;

const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const BLOCK_SIZE: usize = 32; // Cache-friendly block size

pub fn dot_product_tensor(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
    const nDimT1 = t1.shape.len;
    const nDimT2 = t2.shape.len;
    if (nDimT1 != nDimT2) return TensorMathError.InputTensorDifferentShape;
    if (t1.shape[nDimT1 - 1] != t2.shape[nDimT1 - 2]) return TensorMathError.InputTensorsWrongShape;

    if (@TypeOf(outputType) == @TypeOf(inputType)) {
        // Skip check if same type
    } else {
        if (@bitSizeOf(outputType) <= 16) {
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else {
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    const allocator = pkg_allocator;
    var out_shape = try allocator.alloc(usize, nDimT1);
    defer allocator.free(out_shape);

    var total_outer_iterations: usize = 1;
    for (0..(nDimT1 - 2)) |i| {
        out_shape[i] = t1.shape[i];
        total_outer_iterations *= t1.shape[i];
    }
    out_shape[nDimT1 - 2] = t1.shape[nDimT1 - 2];
    out_shape[nDimT1 - 1] = t2.shape[nDimT1 - 1];

    const M = t1.shape[nDimT1 - 2]; // Output rows
    const N = t2.shape[nDimT1 - 1]; // Output cols
    const K = t1.shape[nDimT1 - 1]; // Inner dimension

    var out_tensor = try Tensor(outputType).fromShape(&allocator, out_shape);
    errdefer out_tensor.deinit();

    // Initialize output to zero
    @memset(out_tensor.data, 0);

    // Block sizes for tiling
    const BM = BLOCK_SIZE;
    const BN = BLOCK_SIZE;
    const BK = BLOCK_SIZE;

    // SIMD vector type for fused operations
    const Vec = @Vector(DEFAULT_VECTOR_WIDTH, inputType);

    // Outer blocks
    var i: usize = 0;
    while (i < M) : (i += BM) {
        const i_end = @min(i + BM, M);

        var j: usize = 0;
        while (j < N) : (j += BN) {
            const j_end = @min(j + BN, N);

            var k: usize = 0;
            while (k < K) : (k += BK) {
                const k_end = @min(k + BK, K);

                // Process block
                var ii: usize = i;
                while (ii < i_end) : (ii += 1) {
                    var jj: usize = j;
                    while (jj + DEFAULT_VECTOR_WIDTH <= j_end) : (jj += DEFAULT_VECTOR_WIDTH) {
                        var acc: Vec = @splat(0);

                        // Inner block with SIMD
                        var kk: usize = k;
                        while (kk < k_end) : (kk += 1) {
                            const a = t1.data[ii * K + kk];
                            const b_vec = blk: {
                                var vec: Vec = undefined;
                                for (0..DEFAULT_VECTOR_WIDTH) |v| {
                                    vec[v] = t2.data[kk * N + (jj + v)];
                                }
                                break :blk vec;
                            };
                            // Fused multiply-add
                            acc += @as(Vec, @splat(a)) * b_vec;
                        }

                        // Store result
                        for (0..DEFAULT_VECTOR_WIDTH) |v| {
                            out_tensor.data[ii * N + (jj + v)] += @as(outputType, acc[v]);
                        }
                    }

                    // Handle remaining columns
                    while (jj < j_end) : (jj += 1) {
                        var sum: inputType = 0;
                        var kk: usize = k;
                        while (kk < k_end) : (kk += 1) {
                            sum += t1.data[ii * K + kk] * t2.data[kk * N + jj];
                        }
                        out_tensor.data[ii * N + jj] += @as(outputType, sum);
                    }
                }
            }
        }
    }

    return out_tensor;
}

/// Function that performs the multiplication of two tensors used in a recursive way to handle multidimensional tensors
fn multidim_multiplication(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), t3: *Tensor(outputType), current_depth: usize, location: []usize) !void {
    if (current_depth == (t1.shape.len - 2)) {

        //declaring sum
        var sum: outputType = 0;

        //with the first two for loop I iterate over t3
        for (0..t1.shape[current_depth]) |row| { //for each row of t1

            for (0..t2.shape[current_depth + 1]) |col| { //for each col of t2

                sum = 0;

                for (0..t1.shape[current_depth + 1]) |i| {

                    //compose the location on t1
                    location[t1.shape.len - 1] = i; //location
                    location[t1.shape.len - 2] = row; //location

                    //getting the correct numbers in t1
                    const a = try t1.get_at(location);

                    //compose the location on t2
                    location[t1.shape.len - 1] = col; //location
                    location[t1.shape.len - 2] = i; //location

                    //getting the correct numbers in t2
                    const b = try t2.get_at(location);

                    sum += a * b;
                }

                //compose the location on t3
                location[t1.shape.len - 1] = col; //col on the out tensor matrix
                location[t1.shape.len - 2] = row; //row on the out tensor matrix

                try t3.set_at(location, sum);
            }
        }
    } else {
        for (0..t1.shape[current_depth]) |element_at_current_depth| {
            //print location:
            //std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
            location[current_depth] = element_at_current_depth;
            //otherwise I have to go deeper
            try multidim_multiplication(
                inputType,
                outputType,
                t1,
                t2,
                t3,
                current_depth + 1,
                location,
            );
        }
    }
}
