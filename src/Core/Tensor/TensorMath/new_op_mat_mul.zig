const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

const ArchitectureError = zant.utils.error_handler.ArchitectureError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

// TODO: fetch using zig compiler the size of cache block for target architecture
const CACHE_BLOCK_SIZE_BYTES: usize = 128;

const VEC_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;

// TODO add support for matrix multiplication for matrix distribuited in multi-batch/multi-channel tensors (for example of shape {2, 3, 5, 5}), now supports only tensors with shape {1, 1, N, M}
/// Performs classic matrix multiplication on given tensors using the least 2 dimensions
pub inline fn mat_mul(
    comptime T: anytype, 
    A: *const Tensor(T), 
    B: *const Tensor(T)
    ) !Tensor(T) {

    // The two tensors needs to have the same dimensions N
    if (A.shape.len != B.shape.len) {
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;

    // The last dimension (number of cols) of A must be equal to the second last dimension (number of rows) of B
    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        return TensorMathError.InputTensorsWrongShape;
    }

    // The input tensors must have at least 2 dimensions
    if (dim_num < 2) {
        return TensorMathError.InputTensorsWrongShape;
    }

    const a_rows = A.shape[dim_num - 2];
    const b_cols = B.shape[dim_num - 1];
    const a_cols = A.shape[dim_num - 1];

    if (a_rows * b_cols == 0 or a_cols == 0) {
        return TensorMathError.InputTensorsWrongShape;
    }

    var out_shape = try pkg_allocator.alloc(usize, dim_num);
    defer pkg_allocator.free(out_shape);
    errdefer pkg_allocator.free(out_shape);

    // Copy all dimensions except the last two
    for (0..(dim_num - 2)) |i| {
        out_shape[i] = A.shape[i];
    }

    // Set the last two dimensions to the dimensions of the input tensors
    out_shape[dim_num - 2] = A.shape[dim_num - 2];
    out_shape[dim_num - 1] = B.shape[dim_num - 1];

    // Create output tensor

    var Y = try Tensor(T).fromShape(&pkg_allocator, out_shape);
    errdefer Y.deinit();

    @memset(Y.data, 0);

    try lean_mat_mul(T, A, B, &Y);

    return Y;
}

pub inline fn lean_mat_mul(
    comptime T: anytype, 
    A: *const Tensor(T), 
    B: *const Tensor(T), 
    C: *const Tensor(T)
    ) !void {
    
    // https://coffeebeforearch.github.io/2020/06/23/mmul.html
        
    // A
    //  n colonne
    //  k righe
    
    // B
    //  m colonne
    //  n righe
    
    // C
    //  m colonne
    //  k righe
    
        
    // 000 00 = 00
    // 000 00   00
    // 000 00   00
    // 000      00
    
    const cache_block_size = CACHE_BLOCK_SIZE_BYTES / @sizeOf(T);

    const a_rows = A.shape[A.shape.len-2];
    const a_cols = A.shape[A.shape.len-1];
    
    const b_cols = B.shape[B.shape.len-1];
    //const b_rows = a_cols;
    
    const c_rows = a_rows;
    const c_cols = b_cols;
    
    var c_column_chunk: usize = 0;
    var c_chunk_rows: usize = 0;
    var tile: usize = 0;

    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const C_ptr = C.data.ptr;
    
    //Assumes both dimensions are dim mod(cache_block_size) equivalent.
    while(c_column_chunk < c_cols) : (c_column_chunk+=cache_block_size){
        while(c_chunk_rows < c_rows) : (c_chunk_rows+=1) {
            while(tile<a_rows) : (tile+=cache_block_size) {
                for(0..cache_block_size) |t_row| {
                    for(0..cache_block_size) |t_col| {
                        //std.debug.print("C indices: c_chunk_rows: {d}, c_column_chunk: {d}, t_row: {d}, t_col: {d}\n", .{c_chunk_rows, c_column_chunk, t_row, t_col});
                        //std.debug.print("Calculating: tile: {d}, b_cols: {d}, t_row: {d}, c_column_chunk: {d}, t_col: {d} \n", .{tile, b_cols, t_row, c_column_chunk, t_col});
                        C_ptr[c_chunk_rows * c_cols + c_column_chunk + t_col] +=
                            //For each chunk of b we take a single row of a, that we access in a linear and ordered fashion, going down the column for each new row of c and going along the line for each tile of B
                            A_ptr[c_chunk_rows*a_cols + tile + t_row] *
                            //Each tile is a *cache_block_size* amount of columns apart, each row is another b_cols amount apart, we select the actual row by also specifying the column chunk and column we are acting on
                            B_ptr[tile*b_cols + t_row*b_cols + c_column_chunk + t_col];
                    }
                }
            }
            tile = 0;
        }
        c_chunk_rows = 0;
    }
    
}
