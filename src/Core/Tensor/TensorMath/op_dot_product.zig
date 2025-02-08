const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const assert = std.debug.assert;

const ArchitectureError = @import("errorHandler").ArchitectureError;
const TensorMathError = @import("errorHandler").TensorMathError;

const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const BLOCK_SIZE: usize = 32; // Cache-friendly block size

pub inline fn dot_product_tensor(comptime inputType: type, comptime outputType: type, t1: *const Tensor(inputType), t2: *const Tensor(inputType)) !Tensor(outputType) {
    // Verifica che il numero di dimensioni coincidano e le forme siano compatibili
    const nDimT1 = t1.shape.len;
    const nDimT2 = t2.shape.len;
    if (nDimT1 != nDimT2)
        return TensorMathError.InputTensorDifferentShape;
    if (t1.shape[nDimT1 - 1] != t2.shape[nDimT1 - 2])
        return TensorMathError.InputTensorsWrongShape;

    // Controllo sul tipo di output: se la conversione non è "sicura" (es. da f32 a f16)
    if (@TypeOf(outputType) == @TypeOf(inputType)) {
        // Nessun controllo necessario se sono lo stesso tipo
    } else {
        if (@bitSizeOf(outputType) <= 16) {
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2))
                return TensorMathError.TooSmallOutputType;
        } else {
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType))
                return TensorMathError.TooSmallOutputType;
        }
    }

    const allocator = pkg_allocator;
    var out_shape = try allocator.alloc(usize, nDimT1);
    defer allocator.free(out_shape);

    // Costruisco la forma dell'output:
    // Copio le dimensioni esterne (se presenti) e imposto le ultime due dimensioni in base alla moltiplicazione
    for (0..(nDimT1 - 2)) |i| {
        out_shape[i] = t1.shape[i];
    }
    out_shape[nDimT1 - 2] = t1.shape[nDimT1 - 2];
    out_shape[nDimT1 - 1] = t2.shape[nDimT1 - 1];

    const M = t1.shape[nDimT1 - 2]; // Numero di righe dell'output
    const N = t2.shape[nDimT1 - 1]; // Numero di colonne dell'output
    const K = t1.shape[nDimT1 - 1]; // Dimensione interna (dot product lungo k)

    var out_tensor = try Tensor(outputType).fromShape(&allocator, out_shape);
    errdefer out_tensor.deinit();

    // Inizializza la memoria dell'output a zero
    @memset(out_tensor.data, 0);

    // Setup SIMD
    const vector_width = DEFAULT_VECTOR_WIDTH;
    const Vec = @Vector(vector_width, inputType);
    const HigherVec = @Vector(vector_width, outputType);

    const vec_len = K / vector_width;
    const vec_rem = K % vector_width;

    // Disabilitiamo alcuni controlli per massimizzare le prestazioni
    @setRuntimeSafety(false);

    // Calcola ogni elemento dell'output:
    //   out[i, j] = sum_{k=0}^{K-1} ( t1[i, k] * t2[k, j] )
    var i: usize = 0;
    while (i < M) : (i += 1) {
        var j: usize = 0;
        while (j < N) : (j += 1) {
            // Accumulatore per il dot product in formato SIMD (vettore di dimensione vector_width)
            var sum_vec: HigherVec = @splat(0);

            // Elaborazione vettorializzata lungo la dimensione k
            var k_chunk: usize = 0;
            while (k_chunk < vec_len) : (k_chunk += 1) {
                const offset = k_chunk * vector_width;
                var a_vec: Vec = undefined;
                var b_vec: Vec = undefined;
                // Carica un blocco di vector_width elementi da t1 e t2:
                for (0..vector_width) |v| {
                    a_vec[v] = t1.data[i * K + offset + v];
                    // Per k = offset+v si legge t2[k, j]:
                    b_vec[v] = t2.data[(offset + v) * N + j];
                }
                // Conversione esplicita in outputType per maggiore precisione
                var higher_a: HigherVec = undefined;
                var higher_b: HigherVec = undefined;
                for (0..vector_width) |v| {
                    higher_a[v] = @as(outputType, (a_vec[v]));
                    higher_b[v] = @as(outputType, (b_vec[v]));
                }
                sum_vec += higher_a * higher_b;
            }

            // Riduzione orizzontale del vettore SIMD in uno scalare
            var dot: outputType = 0;
            for (0..vector_width) |v| {
                dot += sum_vec[v];
            }

            // Gestione degli elementi rimanenti lungo k (se K non è multiplo di vector_width)
            const rem_offset = vec_len * vector_width;
            var r: usize = 0;
            while (r < vec_rem) : (r += 1) {
                const a_val = @as(outputType, (t1.data[i * K + rem_offset + r]));
                const b_val = @as(outputType, (t2.data[(rem_offset + r) * N + j]));
                dot += a_val * b_val;
            }

            out_tensor.data[i * N + j] = dot;
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

pub fn benchmark_dot_product() !void {
    const allocator = pkg_allocator;

    // Create two large tensors
    var shape1 = [_]usize{ 1024, 1024 };
    var shape2 = [_]usize{ 1024, 1024 };

    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);
    defer t1.deinit();
    defer t2.deinit();

    // Fill with random data
    for (t1.data, 0..) |_, i| {
        t1.data[i] = @floatFromInt(i % 10);
        t2.data[i] = @floatFromInt(i % 10);
    }

    // Benchmark SIMD version
    const timer = try std.time.Timer.start();
    var result1 = try dot_product_tensor(f32, f32, &t1, &t2);
    defer result1.deinit();
    const simd_time = timer.lap();

    // Benchmark flat version
    const timer2 = try std.time.Timer.start();
    var result2 = try dot_product_tensor_flat(f32, f32, &t1, &t2);
    defer result2.deinit();
    const flat_time = timer2.lap();

    // Benchmark recursive version
    var shape_out = [_]usize{ 1024, 1024 };
    var result3 = try Tensor(f32).fromShape(&allocator, &shape_out);
    defer result3.deinit();
    const location = try allocator.alloc(usize, 2);
    defer allocator.free(location);

    const timer3 = try std.time.Timer.start();
    try multidim_multiplication(f32, f32, &t1, &t2, &result3, 0, location);
    const recursive_time = timer3.lap();

    // Print results
    std.debug.print("\nBenchmark Results:\n", .{});
    std.debug.print("SIMD version: {d:.2} ms\n", .{@as(f64, @floatFromInt(simd_time)) / 1_000_000.0});
    std.debug.print("Flat version: {d:.2} ms\n", .{@as(f64, @floatFromInt(flat_time)) / 1_000_000.0});
    std.debug.print("Recursive version: {d:.2} ms\n", .{@as(f64, @floatFromInt(recursive_time)) / 1_000_000.0});
    std.debug.print("\nSpeedups:\n", .{});
    std.debug.print("SIMD vs Recursive: {d:.2}x\n", .{@as(f64, @floatFromInt(recursive_time)) / @as(f64, @floatFromInt(simd_time))});
    std.debug.print("Flat vs Recursive: {d:.2}x\n", .{@as(f64, @floatFromInt(recursive_time)) / @as(f64, @floatFromInt(flat_time))});
    std.debug.print("SIMD vs Flat: {d:.2}x\n", .{@as(f64, @floatFromInt(flat_time)) / @as(f64, @floatFromInt(simd_time))});

    // Verify results are the same
    for (result1.data, result2.data, result3.data) |v1, v2, v3| {
        if (@abs(v1 - v2) > 0.001 or @abs(v1 - v3) > 0.001) {
            std.debug.print("Warning: Results differ!\n", .{});
            break;
        }
    }
}

/// Implementation of dot product using flat iteration
pub fn dot_product_tensor_flat(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
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
    total_outer_iterations *= t1.shape[nDimT1 - 2] * t2.shape[nDimT1 - 1];

    var out_tensor = try Tensor(outputType).fromShape(&allocator, out_shape);
    errdefer out_tensor.deinit();

    const inner_dim = t1.shape[nDimT1 - 1];
    const t1_stride = t1.shape[nDimT1 - 1];
    const t2_stride = t2.shape[nDimT1 - 1];
    const out_stride = out_tensor.shape[nDimT1 - 1];

    var batch_idx: usize = 0;
    while (batch_idx < total_outer_iterations) : (batch_idx += 1) {
        const out_row = (batch_idx / out_stride) % out_tensor.shape[nDimT1 - 2];
        const out_col = batch_idx % out_stride;

        var sum: outputType = 0;
        const row_offset = out_row * t1_stride;
        const col_offset = out_col;

        var k: usize = 0;
        while (k < inner_dim) : (k += 1) {
            const t1_val = t1.data[row_offset + k];
            const t2_val = t2.data[k * t2_stride + col_offset];
            sum += t1_val * t2_val;
        }

        out_tensor.data[batch_idx] = sum;
    }

    return out_tensor;
}
