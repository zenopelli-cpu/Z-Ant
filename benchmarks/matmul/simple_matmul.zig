const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

pub inline fn lean_mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), C: *const Tensor(T)) !void {
    const a_rows = A.shape[A.shape.len - 2];
    const a_cols = A.shape[A.shape.len - 1];

    const b_cols = B.shape[B.shape.len - 1];

    const c_rows = a_rows;
    const c_cols = b_cols;

    const A_ptr = A.data.ptr;
    const B_ptr = B.data.ptr;
    const C_ptr = C.data.ptr;

    for (0..c_rows) |row| {
        for (0..c_cols) |col| {
            for (0..a_cols) |idx| {
                C_ptr[row * c_cols + col] += A_ptr[row * a_cols + idx] * B_ptr[idx * c_cols + col];
            }
        }
    }
}
