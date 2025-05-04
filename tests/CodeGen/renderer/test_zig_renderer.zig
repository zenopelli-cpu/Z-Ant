const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const Renderer = codegen.renderer;
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const Tensor = zant.core.tensor.Tensor;

const sum_tensors = zant.core.tensor.math_standard.sum_tensors;

const ZigRenderer = Renderer.ZigRenderer;
const lowerAdd = zant.core.tensor.math_standard.lowerAdd;
const lowerMatMul = zant.core.tensor.math_standard.lowerMatMul;
const UOpBuilder = zant.uops.UOpBuilder;
const DType = zant.uops.DType;

// /* REMOVED OLD TESTS
// test "Arithmetic operations" { ... }
// ... etc ...
// test "Rendering memory operations" { ... }
// */

test "LowerAdd Pipeline" {
    std.debug.print("Running zig renderer lowerAdd pipeline test\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    // No defer needed if we don't take ownership of slice

    // 2. Define inputs for lowerAdd (example shapes/strides)
    const A_id: usize = 0; // Simulated input tensor ID
    const B_id: usize = 1; // Simulated input tensor ID
    const out_shape = &.{ 2, 3 }; // Example output shape
    const strideA = &.{ 3, 1 }; // Example strides for A (row-major)
    const strideB = &.{ 0, 1 }; // Example strides for B (broadcast dim 0)
    const out_dtype = DType.f32;

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = lowerAdd(
        &builder,
        A_id,
        B_id,
        out_shape,
        strideA,
        strideB,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    // Deinit builder immediately as its list is now empty
    builder.deinit();
    // Defer freeing the main slice AFTER freeing internal src slices
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {}; // Dump to stderr
    }
    std.debug.print("--------------------\n", .{});

    // Add defer to free duplicated src slices within the owned list
    defer {
        std.debug.print("DEBUG: Freeing internal src for {d} uops in test\n", .{uops_list.len});
        for (uops_list) |uop| {
            // Free src (only if non-empty)
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            // Free duplicated arg payloads (only if non-null and relevant type)
            if (uop.arg) |arg_val| {
                // Use switch for type-safe union payload access
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            // Only free if non-empty
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {}, // VIEW op with unexpected arg type? Ignore.
                    }
                }
                // Add else if for other duplicated args
                // else if (uop.op == .SOME_OTHER_OP) { ... }
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit(); // Deinit renderer AFTER use

    // Specify which IDs are inputs
    const input_ids = &[_]usize{ A_id, B_id };
    try renderer.render_as_function(uops_list, input_ids); // Call the new method

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function ---\n{s}\n---------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/loweradd_output_function.zig"; // Save inside tests dir
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true }); // Ensure write permissions
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated function saved to {s}\n", .{output_filename});

    // Optional: Clean up the generated file
    // try std.fs.cwd().deleteFile(output_filename);
}

test "LowerMatMul Pipeline" {
    std.debug.print("Running zig renderer lowerMatMul pipeline test\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    // No defer needed if we don't take ownership of slice

    // 2. Define inputs for lowerMatMul
    const A_id: usize = 0; // Simulated input tensor ID (Matrix A)
    const B_id: usize = 1; // Simulated input tensor ID (Matrix B)

    // Example: C[M, N] = A[M, K] @ B[K, N]
    // Let M=2, K=2, N=3
    const M: usize = 2;
    const K: usize = 2;
    const N: usize = 3;

    const shapeA = &.{ M, K }; // Shape of A
    const shapeB = &.{ K, N }; // Shape of B
    const out_shape = &.{ M, N }; // Shape of C (output)

    const out_dtype = DType.f32;

    // 3. Call lowerMatMul to generate UOps
    const out_buf_id = lowerMatMul(
        &builder,
        A_id,
        B_id,
        shapeA,
        shapeB,
        out_shape,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    // Deinit builder immediately as its list is now empty
    builder.deinit();
    // Defer freeing the main slice AFTER freeing internal src slices
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps (MatMul) ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {}; // Dump to stderr
    }
    std.debug.print("-----------------------------\n", .{});

    // Add defer to free duplicated src slices within the owned list
    defer {
        std.debug.print("DEBUG: Freeing internal src for {d} uops in MatMul test\n", .{uops_list.len});
        for (uops_list) |uop| {
            // Free src (only if non-empty)
            if (uop.src.len > 0) {
                allocator.free(@constCast(uop.src));
            }
            // Free duplicated arg payloads (only if non-null and relevant type)
            if (uop.arg) |arg_val| {
                // Use switch for type-safe union payload access
                if (uop.op == .VIEW) {
                    switch (arg_val) {
                        .view_meta => |vm| {
                            // Only free if non-empty
                            if (vm.shape.len > 0) allocator.free(@constCast(vm.shape));
                            if (vm.strides.len > 0) allocator.free(@constCast(vm.strides));
                        },
                        else => {}, // VIEW op with unexpected arg type? Ignore.
                    }
                }
                // Add else if for other duplicated args
                // else if (uop.op == .SOME_OTHER_OP) { ... }
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit(); // Deinit renderer AFTER use

    // Specify which IDs are inputs
    const input_ids = &[_]usize{ A_id, B_id };
    try renderer.render_as_function(uops_list, input_ids); // Call the new method

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function (MatMul) ---\n{s}\n---------------------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/lower_matmul_output_function.zig"; // Save inside tests dir
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true }); // Ensure write permissions
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated matmul function saved to {s}\n", .{output_filename});

    // Optional: Clean up the generated file
    // try std.fs.cwd().deleteFile(output_filename);
}
test "Test Generated LowerMatMul Kernel" {
    std.debug.print("Testing generated kernel from lower_matmul_output_function.zig\n", .{});
    const allocator = std.testing.allocator;
    const kernel = @import("lower_matmul_output_function.zig"); // Import the generated file

    // 1. Define input data (A[M, K], B[K, N])
    const M = 2;
    const K = 2;
    const N = 3;
    const input_data_0: [M * K]f32 = .{ // A: 2x2
        1.0, 2.0, // Row 0
        3.0, 4.0, // Row 1
    };
    const input_data_1: [K * N]f32 = .{ // B: 2x3
        5.0, 6.0, 7.0, // Row 0
        8.0, 9.0, 10.0, // Row 1
    };

    // 2. Call the generated kernel
    const result_slice = try kernel.generated_kernel(allocator, &input_data_0, &input_data_1);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size M*N)

    // 3. Define expected output C[M, N]
    const expected_result: [M * N]f32 = .{
        21.0, 24.0, 27.0, // Row 0
        47.0, 54.0, 61.0, // Row 1
    };

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, &expected_result, result_slice);

    std.debug.print("Generated matmul kernel test passed!\n", .{});
}

test "Test Generated LowerAdd Kernel" {
    std.debug.print("Testing generated kernel from loweradd_output_function.zig\n", .{});
    const allocator = std.testing.allocator;
    // Import the generated kernel file
    const kernel = @import("loweradd_output_function.zig");

    // 1. Define input data based on LowerAdd Pipeline shapes/strides
    // A shape {2, 3}, stride {3, 1} -> 6 elements, row-major
    const input_data_0 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    // B shape {2, 3}, stride {0, 1} -> Broadcast dim 0, reads elements {10, 11, 12} repeatedly
    // Kernel expects a slice, provide the full data based on loop access pattern.
    // The generated kernel from LowerAdd likely loops 0..6.
    // Based on stride {0, 1}, B[0, j] = B_data[j], B[1, j] = B_data[j]
    // If the kernel accesses linearly (0..6), it accesses:
    // idx 0 -> A[0,0], B[0,0] -> input_0[0], input_1[0]
    // idx 1 -> A[0,1], B[0,1] -> input_0[1], input_1[1]
    // idx 2 -> A[0,2], B[0,2] -> input_0[2], input_1[2]
    // idx 3 -> A[1,0], B[1,0] -> input_0[3], input_1[0]  <-- B repeats
    // idx 4 -> A[1,1], B[1,1] -> input_0[4], input_1[1]  <-- B repeats
    // idx 5 -> A[1,2], B[1,2] -> input_0[5], input_1[2]  <-- B repeats
    // So input_1 needs to contain {10, 11, 12} for the generated loop accesses.
    // However, the generated code *might* pass the full slice. Let's assume it takes the broadcasted view:
    const input_data_1 = [_]f32{ 10.0, 11.0, 12.0 }; // Only the unique values due to broadcast stride {0,1}
    // IMPORTANT: The generated kernel likely accesses input_1 linearly based on the RANGE(0..6)
    // It will calculate addresses based on strides {3,1} for input_0 and {0,1} for input_1
    // GEP input_0 (View strides {3,1}): (idx_3 / 3) * 3 + (idx_3 % 3) * 1 = idx_3
    // GEP input_1 (View strides {0,1}): (idx_3 / 3) * 0 + (idx_3 % 3) * 1 = idx_3 % 3
    // So, input_1 needs 3 elements, accessed via index % 3.

    // 2. Call the generated kernel
    // Signature: pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32, input_1: []const f32) ![]f32
    const result_slice = try kernel.generated_kernel(allocator, &input_data_0, &input_data_1);
    defer allocator.free(result_slice); // Kernel allocates the output slice (size 6)

    // 3. Define expected output
    // C[i] = A[i] + B[i % 3]
    const expected_result = [_]f32{
        1.0 + 10.0, // 11.0
        2.0 + 11.0, // 13.0
        3.0 + 12.0, // 15.0
        4.0 + 10.0, // 14.0
        5.0 + 11.0, // 16.0
        6.0 + 12.0, // 18.0
    };

    // 4. Compare results
    try std.testing.expectEqualSlices(f32, &expected_result, result_slice);

    std.debug.print("Generated LowerAdd kernel test passed!\n", .{});
}
