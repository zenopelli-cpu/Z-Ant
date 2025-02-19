const std = @import("std");
const Tensor = @import("tensor").Tensor;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;
const pkg_allocator = @import("pkgAllocator").allocator;
const TensMath = @import("tensor_math_standard.zig");
const LeanTensMath = @import("tensor_math_lean.zig");

// Note that this functions needs SIMD optimizations

/// Implements the GEMM operator from the ONNX standard https://onnx.ai/onnx/operators/onnx__Gemm.html Y = alpha*A*B + beta*C
/// As for now the broadcasting only accours in rows and cols dimensions, while batch and channel dimensions must have the same size
pub fn gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: f32, beta: f32, transA: bool, transB: bool) !Tensor(T) {
    var cond_A: usize = 0;
    var cond_B: usize = 0;
    var res_rows: usize = 0;
    var res_cols: usize = 0;

    // verifying correct shape, batch, channels
    if (A.shape.len != 4 or B.shape.len != 4) {
        return TensorMathError.InputTensorsWrongShape;
    }
    if (A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]) {
        return TensorMathError.InputTensorDifferentShape;
    }

    // verifying matrix multiplication viability, getting result shape
    if (transA) {
        cond_A = A.shape[2];
        res_rows = A.shape[3];
    } else {
        cond_A = A.shape[3];
        res_rows = A.shape[2];
    }

    if (transB) {
        cond_B = B.shape[3];
        res_cols = B.shape[2];
    } else {
        cond_B = B.shape[2];
        res_cols = B.shape[3];
    }

    if (cond_A != cond_B) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // control on optional tensor C unwrapping it
    if (C) |actual_C| {
        if (actual_C.shape.len != 4) {
            return TensorMathError.InputTensorsWrongShape;
        }
        if (actual_C.shape[0] != A.shape[0] or actual_C.shape[1] != A.shape[1]) {
            return TensorMathError.InputTensorDifferentShape;
        }
        if ((actual_C.shape[2] != res_rows and actual_C.shape[2] != 1) or (actual_C.shape[3] != res_cols and actual_C.shape[3] != 1)) {
            return TensorMathError.IncompatibleBroadcastShapes;
        }
    }

    // creating result tensor
    var res_shape = try pkg_allocator.alloc(usize, 4);
    defer pkg_allocator.free(res_shape);
    res_shape[0] = A.shape[0];
    res_shape[1] = A.shape[1];
    res_shape[2] = res_rows;
    res_shape[3] = res_cols;

    var result = try Tensor(T).fromShape(&pkg_allocator, res_shape);
    errdefer result.deinit();

    // debug
    // for (0..A.data.len) |i|
    //     std.debug.print("\n gemm A[{d}] {d}", .{ i, A.data[i] });
    // for (0..B.data.len) |i|
    //     std.debug.print("\n gemm B[{d}] {d}", .{ i, B.data[i] });
    // if (C) |CC| {
    //     for (0..CC.data.len) |i|
    //         std.debug.print("\n gemm C[{d}] {d}", .{ i, CC.data[i] });
    // }

    try lean_gemm(T, A, B, C, alpha, beta, transA, transB, &result);

    return result;
}

/// Lean version of gemm, output Tensor must be preconstructed and 0 filled
pub fn lean_gemm(comptime T: anytype, A: *Tensor(T), B: *Tensor(T), C: ?*Tensor(T), alpha: f32, beta: f32, transA: bool, transB: bool, result: *Tensor(T)) !void {
    var actual_A_ptr = A;
    var actual_B_ptr = B;

    // applying transposition
    if (transA) {
        var actual_A = try transposeLastTwo(T, A);
        actual_A_ptr = &actual_A;
    }
    if (transB) {
        var actual_B = try transposeLastTwo(T, B);
        actual_B_ptr = &actual_B;
    }

    // debug
    // for (0..actual_A_ptr.data.len) |i|
    //     std.debug.print("\n TRANS LEAN A[{d}] {d}", .{ i, actual_A_ptr.data[i] });
    // for (0..actual_B_ptr.data.len) |i|
    //     std.debug.print("\n TRANS LEAN B[{d}] {d}", .{ i, actual_B_ptr.data[i] });
    // if (C) |CC| {
    //     for (0..CC.data.len) |i|
    //         std.debug.print("\n NO TRANS LEAN C[{d}] {d}", .{ i, CC.data[i] });
    // }

    // result = alpha * A * B
    try LeanTensMath.lean_dot_product_tensor(T, T, actual_A_ptr, actual_B_ptr, result);
    for (0..result.size) |i| {
        result.data[i] *= alpha;
    }

    // debug
    // for (0..result.data.len) |i|
    //     std.debug.print("\n LEAN PRODUCT PR[{d}] {d}", .{ i, result.data[i] });

    // result = result + beta * C
    if (C) |actual_C_ptr| {
        if (beta != 0) {

            // no broadcast necessary
            if (result.size == actual_C_ptr.size) {
                for (0..result.size) |i| {
                    result.data[i] += actual_C_ptr.data[i] * beta;
                }
            }

            // broadcast from C to result
            else {
                const res_rows = result.shape[result.shape.len - 2];
                const res_cols = result.shape[result.shape.len - 1];

                // Determine whether broadcast on rows and cols is needed or not
                const c_rows = if (actual_C_ptr.shape.len >= 2) actual_C_ptr.shape[actual_C_ptr.shape.len - 2] else 1;
                const c_cols = if (actual_C_ptr.shape.len >= 1) actual_C_ptr.shape[actual_C_ptr.shape.len - 1] else 1;

                for (0..actual_A_ptr.shape[0]) |b| {
                    for (0..actual_A_ptr.shape[1]) |c| {
                        for (0..res_rows) |i| {
                            for (0..res_cols) |j| {

                                // index used on C
                                const ci = if (c_rows == 1) 0 else i;
                                const cj = if (c_cols == 1) 0 else j;

                                // summing the product of C and beta
                                const result_index = (((b * result.shape[1]) + c) * result.shape[2] + i) * result.shape[3] + j;
                                const c_index = (((b * actual_C_ptr.shape[1]) + c) * actual_C_ptr.shape[2] + ci) * actual_C_ptr.shape[3] + cj;
                                result.data[result_index] += actual_C_ptr.data[c_index] * beta;
                            }
                        }
                    }
                }
            }
        }
    }

    // debug
    // for (0..result.data.len) |i|
    //     std.debug.print("\n LEAN SUM SM[{d}] {d}", .{ i, result.data[i] });

    if (transA) {
        actual_A_ptr.deinit();
    }
    if (transB) {
        actual_B_ptr.deinit();
    }
}

// TODO: move it in lib_shape_math.zig, add test
/// Given a 4D tensor it returns the tensor with the last 2 dimensions transposed. Operates on both data and shape, does not modify self, used by gemm.
pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {

    // Veryfing correct shape
    if (tensor.shape.len != 4) {
        return TensorMathError.InputTensorsWrongShape;
    }

    const batch = tensor.shape[0];
    const channel = tensor.shape[1];
    const rows = tensor.shape[2];
    const cols = tensor.shape[3];
    const total = batch * channel * rows * cols;

    // New shape
    const newShape = try pkg_allocator.alloc(usize, 4);
    errdefer pkg_allocator.free(newShape);
    newShape[0] = batch;
    newShape[1] = channel;
    newShape[2] = cols;
    newShape[3] = rows;

    // New data
    const outData = try tensor.allocator.alloc(T, total);
    errdefer tensor.allocator.free(outData);

    // Traspose the elements within the matrix
    for (0..batch) |b| {
        for (0..channel) |c| {
            for (0..rows) |i| {
                for (0..cols) |j| {
                    const index_in = (((b * channel) + c) * rows + i) * cols + j;
                    const index_out = (((b * channel) + c) * cols + j) * rows + i;
                    outData[index_out] = tensor.data[index_in];
                }
            }
        }
    }

    // Build tensor and return
    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = tensor.allocator,
    };
}
