const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

test "pow f64 - basic case" {
    std.debug.print("\n     test: onehot_standard f64 - basic case", .{});
    const allocator = pkgAllocator.allocator;

    var indices_array = [_]f64{ 0, 1 };
    var indices_shape = [_]usize{2};

    var values_array = [_]f64{ 0.0, 10.0 };
    var values_shape = [_]usize{2};

    var baseTensor = try Tensor(f64).fromArray(&allocator, &indices_array, &indices_shape);
    var expTensor = try Tensor(f64).fromArray(&allocator, &values_array, &values_shape);

    defer baseTensor.deinit();
    defer expTensor.deinit();

    var result = try TensMath.pow(f64, &baseTensor, &expTensor);
    defer result.deinit();

    //check shape, output

}

test "pow broadcast - 5D complex broadcasting" {
    std.debug.print("\n     test: pow broadcast - 5D complex", .{});
    const allocator = pkgAllocator.allocator;

    // Base: 1x3x1x4x2
    var base_array = [_]f32{2.0} ** 24;
    var base_shape = [_]usize{ 1, 3, 1, 4, 2 };

    // Exp: 2x1x5x1x2
    var exp_array = [_]f32{2.0} ** 20;
    var exp_shape = [_]usize{ 2, 1, 5, 1, 2 };

    var baseTensor = try Tensor(f32).fromArray(&allocator, &base_array, &base_shape);
    var expTensor = try Tensor(f32).fromArray(&allocator, &exp_array, &exp_shape);

    defer baseTensor.deinit();
    defer expTensor.deinit();

    var result = try TensMath.pow(f32, &baseTensor, &expTensor);
    defer result.deinit();

    // Output shape dovrebbe essere 2x3x5x4x2
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 5, 4, 2 }, result.shape);
}

test "pow broadcast - complex asymmetric 4D broadcasting with verification" {
    std.debug.print("\n     test: pow broadcast - complex asymmetric 4D", .{});
    const allocator = pkgAllocator.allocator;

    // Base: 3x1x4x1 tensor
    // Valori organizzati per verificare ogni dimensione del broadcast
    var base_array = [_]f32{
        // Batch 0
        2.0, 3.0, 4.0, 5.0,
        // Batch 1
        2.0, 3.0, 4.0, 5.0,
        // Batch 2
        2.0, 3.0, 4.0, 5.0,
    };
    var base_shape = [_]usize{ 3, 1, 4, 1 };

    // Exp: 1x2x1x3 tensor
    // Valori specifici per testare ogni combinazione
    var exp_array = [_]f32{
        // Channel 0
        1.0, 2.0, 0.5,
        // Channel 1
        3.0, 0.0, -1.0,
    };
    var exp_shape = [_]usize{ 1, 2, 1, 3 };

    var baseTensor = try Tensor(f32).fromArray(&allocator, &base_array, &base_shape);
    var expTensor = try Tensor(f32).fromArray(&allocator, &exp_array, &exp_shape);

    defer baseTensor.deinit();
    defer expTensor.deinit();

    var result = try TensMath.pow(f32, &baseTensor, &expTensor);
    defer result.deinit();

    // Output shape: 3x2x4x3 = 72 elementi
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 2, 4, 3 }, result.shape);

    // Verifica manuale di alcuni elementi critici
    // Formula indice: batch*24 + channel*12 + spatial*3 + innermost

    // Batch 0, Channel 0, Spatial 0 (base=2.0):
    // [2^1, 2^2, 2^0.5] = [2.0, 4.0, 1.414...]
    const idx_000 = 0 * 24 + 0 * 12 + 0 * 3;
    try std.testing.expectApproxEqAbs(2.0, result.data[idx_000 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(4.0, result.data[idx_000 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(1.41421356, result.data[idx_000 + 2], 1e-5);

    // Batch 0, Channel 0, Spatial 1 (base=3.0):
    // [3^1, 3^2, 3^0.5] = [3.0, 9.0, 1.732...]
    const idx_001 = 0 * 24 + 0 * 12 + 1 * 3;
    try std.testing.expectApproxEqAbs(3.0, result.data[idx_001 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(9.0, result.data[idx_001 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(1.73205080, result.data[idx_001 + 2], 1e-5);

    // Batch 0, Channel 1, Spatial 0 (base=2.0, exp=[3.0, 0.0, -1.0]):
    // [2^3, 2^0, 2^-1] = [8.0, 1.0, 0.5]
    const idx_010 = 0 * 24 + 1 * 12 + 0 * 3;
    try std.testing.expectApproxEqAbs(8.0, result.data[idx_010 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_010 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(0.5, result.data[idx_010 + 2], 1e-5);

    // Batch 0, Channel 1, Spatial 2 (base=4.0, exp=[3.0, 0.0, -1.0]):
    // [4^3, 4^0, 4^-1] = [64.0, 1.0, 0.25]
    const idx_012 = 0 * 24 + 1 * 12 + 2 * 3;
    try std.testing.expectApproxEqAbs(64.0, result.data[idx_012 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_012 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(0.25, result.data[idx_012 + 2], 1e-5);

    // Batch 1, Channel 0, Spatial 3 (base=5.0, exp=[1.0, 2.0, 0.5]):
    // [5^1, 5^2, 5^0.5] = [5.0, 25.0, 2.236...]
    const idx_103 = 1 * 24 + 0 * 12 + 3 * 3;
    try std.testing.expectApproxEqAbs(5.0, result.data[idx_103 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(25.0, result.data[idx_103 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(2.23606797, result.data[idx_103 + 2], 1e-5);

    // Batch 2, Channel 1, Spatial 1 (base=3.0, exp=[3.0, 0.0, -1.0]):
    // [3^3, 3^0, 3^-1] = [27.0, 1.0, 0.333...]
    const idx_211 = 2 * 24 + 1 * 12 + 1 * 3;
    try std.testing.expectApproxEqAbs(27.0, result.data[idx_211 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_211 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(0.33333333, result.data[idx_211 + 2], 1e-5);

    // Verifica che tutti gli elementi siano stati calcolati (non NaN o Inf)
    for (result.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

test "pow broadcast - complex asymmetric 4D broadcasting with verification 2.0" {
    std.debug.print("\n     test: pow broadcast - complex asymmetric 4D", .{});
    const allocator = pkgAllocator.allocator;

    // Base: 3x1x4x1 tensor
    // Valori organizzati per verificare ogni dimensione del broadcast
    var base_array = [_]f32{
        // Batch 0
        2.0, 3.0, 4.0, 5.0,
        // Batch 1
        2.0, 3.0, 4.0, 5.0,
        // Batch 2
        2.0, 3.0, 4.0, 5.0,
    };
    var base_shape = [_]usize{ 3, 1, 4, 1 };

    // Exp: 1x2x1x3 tensor
    // Valori specifici per testare ogni combinazione
    var exp_array = [_]f32{
        // Channel 0
        1.0, 2.0, 0.5,
        // Channel 1
        3.0, 0.0, -1.0,
    };
    var exp_shape = [_]usize{ 1, 2, 1, 3 };

    var baseTensor = try Tensor(f32).fromArray(&allocator, &base_array, &base_shape);
    var expTensor = try Tensor(f32).fromArray(&allocator, &exp_array, &exp_shape);

    defer baseTensor.deinit();
    defer expTensor.deinit();

    var result = try TensMath.pow(f32, &baseTensor, &expTensor);
    defer result.deinit();

    // Output shape: 3x2x4x3 = 72 elementi
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 2, 4, 3 }, result.shape);

    // Verifica manuale di alcuni elementi critici
    // Formula indice: batch*24 + channel*12 + spatial*3 + innermost

    // Batch 0, Channel 0, Spatial 0 (base=2.0):
    // [2^1, 2^2, 2^0.5] = [2.0, 4.0, 1.414...]
    const idx_000 = 0 * 24 + 0 * 12 + 0 * 3;
    try std.testing.expectApproxEqAbs(2.0, result.data[idx_000 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(4.0, result.data[idx_000 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(1.41421356, result.data[idx_000 + 2], 1e-5);

    // Batch 0, Channel 0, Spatial 1 (base=3.0):
    // [3^1, 3^2, 3^0.5] = [3.0, 9.0, 1.732...]
    const idx_001 = 0 * 24 + 0 * 12 + 1 * 3;
    try std.testing.expectApproxEqAbs(3.0, result.data[idx_001 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(9.0, result.data[idx_001 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(1.73205080, result.data[idx_001 + 2], 1e-5);

    // Batch 0, Channel 1, Spatial 0 (base=2.0, exp=[3.0, 0.0, -1.0]):
    // [2^3, 2^0, 2^-1] = [8.0, 1.0, 0.5]
    const idx_010 = 0 * 24 + 1 * 12 + 0 * 3;
    try std.testing.expectApproxEqAbs(8.0, result.data[idx_010 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_010 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(0.5, result.data[idx_010 + 2], 1e-5);

    // Batch 0, Channel 1, Spatial 2 (base=4.0, exp=[3.0, 0.0, -1.0]):
    // [4^3, 4^0, 4^-1] = [64.0, 1.0, 0.25]
    const idx_012 = 0 * 24 + 1 * 12 + 2 * 3;
    try std.testing.expectApproxEqAbs(64.0, result.data[idx_012 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_012 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(0.25, result.data[idx_012 + 2], 1e-5);

    // Batch 1, Channel 0, Spatial 3 (base=5.0, exp=[1.0, 2.0, 0.5]):
    // [5^1, 5^2, 5^0.5] = [5.0, 25.0, 2.236...]
    const idx_103 = 1 * 24 + 0 * 12 + 3 * 3;
    try std.testing.expectApproxEqAbs(5.0, result.data[idx_103 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(25.0, result.data[idx_103 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(2.23606797, result.data[idx_103 + 2], 1e-5);

    // Batch 2, Channel 1, Spatial 1 (base=3.0, exp=[3.0, 0.0, -1.0]):
    // [3^3, 3^0, 3^-1] = [27.0, 1.0, 0.333...]
    const idx_211 = 2 * 24 + 1 * 12 + 1 * 3;
    try std.testing.expectApproxEqAbs(27.0, result.data[idx_211 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, result.data[idx_211 + 1], 1e-5);
    try std.testing.expectApproxEqAbs(0.33333333, result.data[idx_211 + 2], 1e-5);

    // Verifica che tutti gli elementi siano stati calcolati (non NaN o Inf)
    for (result.data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}
