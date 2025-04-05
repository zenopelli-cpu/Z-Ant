const zant = @import("zant");
const std = @import("std");
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;

const CACHE_BLOCK_SIZE_BYTES: usize = std.atomic.cache_line;

pub inline fn lean_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), C: *const Tensor(T)) !void {

    // https://coffeebeforearch.github.io/2020/06/23/mmul.html

    // A
    //  k righe
    //  n colonne

    // B
    //  n righe
    //  m colonne

    // C
    //  m colonne
    //  k righe
    const VEC_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);

    std.debug.print("Suggested vector length is {any}\n", .{VEC_WIDTH});
    const cache_block_size = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));
    std.debug.print("Cache block size in bytes is {d}\n", .{CACHE_BLOCK_SIZE_BYTES});
    std.debug.print("Cache block size for type is {d}\n", .{cache_block_size});

    const a_rows = A.shape[A.shape.len - 2];
    const a_cols = A.shape[A.shape.len - 1];

    const b_cols = B.shape[B.shape.len - 1];
    //const b_rows = a_cols;

    const c_rows = a_rows;
    const c_cols = b_cols;

    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const C_ptr = C.data.ptr;

    // Determine the number of elements for SIMD vector

    // SIMD Implementation

    const nearest_c_cols = cache_block_size * (c_cols / cache_block_size);
    
    const nearest_a_cols = cache_block_size * (a_cols / cache_block_size);
    const remaining_rows = a_cols - nearest_a_cols;


    var c_chunk_column: usize = 0;
    while (c_chunk_column < nearest_c_cols) : (c_chunk_column += cache_block_size) {
        for (0..c_rows) |c_chunk_row| {
            var tile: usize = 0;

            while (tile < nearest_a_cols) : (tile += cache_block_size) {
                for (0..cache_block_size) |t_row| {
                    simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, tile, t_row, c_chunk_column, c_chunk_row);
                }
            }

            // Handle rows that are not a multiple of cache_block_size
            var last_tile_row: usize = 0;
            while (last_tile_row < remaining_rows) : (last_tile_row += 1) {
                simd_tile_mul(T, A_ptr, a_cols, B_ptr, b_cols, C_ptr, c_cols, nearest_a_cols, last_tile_row, c_chunk_column, c_chunk_row);
            }
        }
    }
}

inline fn tile_mul(
    comptime T: anytype,
    A_ptr: [*]T,
    a_cols: usize,
    B_ptr: [*]T,
    b_cols: usize,
    C_ptr: [*]T,
    c_cols: usize,
    tile: usize,
    t_row: usize,
    c_chunk_column: usize,
    c_chunk_row: usize,
) void {
    const CACHE_BLOCK_SIZE = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));
    const end_col = CACHE_BLOCK_SIZE;

    for (0..end_col) |t_col| {
        C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] +=
            A_ptr[c_chunk_row * a_cols + tile + t_row] *
            B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col];
    }
}

inline fn simd_tile_mul(
    comptime T: anytype,
    A_ptr: [*]T,
    a_cols: usize,
    B_ptr: [*]T,
    b_cols: usize,
    C_ptr: [*]T,
    c_cols: usize,
    tile: usize,
    t_row: usize,
    c_chunk_column: usize,
    c_chunk_row: usize,
) void {
    const CACHE_BLOCK_SIZE = comptime (CACHE_BLOCK_SIZE_BYTES / @sizeOf(T));

    const VEC_WIDTH: usize = comptime (std.simd.suggestVectorLength(T) orelse 4);
    //const VEC_WIDTH: usize = comptime (8);

    // Ensure that c_chunk_column + CACHE_BLOCK_SIZE does not exceed c_cols
    const end_col = CACHE_BLOCK_SIZE;

    const a_val = A_ptr[c_chunk_row * a_cols + tile + t_row];

    // Create a vector filled with the same value of A
    const a_vec: @Vector(VEC_WIDTH, T) = @splat(a_val);
    var b_vec: @Vector(VEC_WIDTH, T) = undefined;
    var c_vec: @Vector(VEC_WIDTH, T) = undefined;

    // Iteration on columns in blocks of simd_lanes
    var t_col: usize = 0;
    while (t_col + VEC_WIDTH <= end_col) : (t_col += VEC_WIDTH) {

        // Load elements of B into a vector
        for (0..VEC_WIDTH) |i| {
            b_vec[i] = B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col + i];
        }

        // Load current values of C
        for (0..VEC_WIDTH) |i| {
            c_vec[i] = C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col + i];
        }

        // Multiply and accumulate
        c_vec += a_vec * b_vec;

        // Write the result in C
        for (0..VEC_WIDTH) |i| {
            C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col + i] = c_vec[i];
        }
    }

    //Handle remaining columns without SIMD
    while (t_col < end_col) : (t_col += 1) {
        C_ptr[c_chunk_row * c_cols + c_chunk_column + t_col] +=
            A_ptr[c_chunk_row * a_cols + tile + t_row] *
            B_ptr[tile * b_cols + t_row * b_cols + c_chunk_column + t_col];
    }
}
