const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

const ArchitectureError = zant.utils.error_handler.ArchitectureError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const CACHE_BLOCK_SIZE_BYTES: usize = 32;

const VEC_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;

// TODO: add support for matrix multiplication for matrix distribuited in multi-batch/multi-channel tensors (for example of shape {2, 3, 5, 5}), now supports only tensors with shape {1, 1, N, M}
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
    
    //const cache_block_size = CACHE_BLOCK_SIZE_BYTES / @sizeOf(T);
    const cache_block_size = 4;

    const a_rows = A.shape[A.shape.len-2];
    const a_cols = A.shape[A.shape.len-1];
    
    const b_cols = B.shape[B.shape.len-1];
    const b_rows = a_cols;
    
    const c_rows = a_rows;
    const c_cols = b_cols;
    
    var col_chunk: usize = 0;
    var row: usize = 0;
    var tile: usize = 0;
    
    //For each chunk in C
    while (col_chunk < b_cols) : (col_chunk+=cache_block_size) {

        while (row < a_rows) : (row+=1) {
            
            while(tile < c_rows) : (tile+=cache_block_size){ 
                //Operate on single tile in column corresponding to the Rows in C
                for(0..cache_block_size)|tile_row|{ 
                    //
                    for(0..cache_block_size)|tile_col|{
                        //Member by member in tile add partial result in corresponding location in C
                        std.debug.print("cache_block_size: {}\n", .{cache_block_size});
                        std.debug.print("tile_row: {}, tile_col: {}\n", .{tile_row, tile_col});
                        std.debug.print("row: {}, tile: {}\n", .{row, tile});
                        
                        C.data[row * c_cols + col_chunk + tile_col] +=
                            A.data[row * a_cols + tile + tile_row] *
                            B.data[tile * b_cols + tile_row * b_rows + col_chunk + tile_col];
                        
                        std.debug.print("---------------- C MATRIX ------------- \n\n", .{});
                        for (0..c_rows) |c_row| {
                            for (0..c_cols) |c_col| {
                                std.debug.print("  {any}  ", .{C.data[c_row * c_cols + c_col]});
                            }
                            std.debug.print("\n", .{});
                        }
                        std.debug.print("----------------------------- \n\n", .{});
                        
                    }
                }
            }
        }
    }
}
