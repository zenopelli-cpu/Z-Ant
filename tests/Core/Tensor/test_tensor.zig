const std = @import("std");
const Tensor = @import("tensor").Tensor;
//import error library
const TensorError = @import("errorHandler").TensorError;
const pkgAllocator = @import("pkgAllocator");

const expect = std.testing.expect;

test {
    _ = @import("TensorMath/test_tensor_math.zig");
}

test "Tensor test description" {
    std.debug.print("\n--- Running tensor tests\n", .{});
}

test "init() test" {
    std.debug.print("\n     test: init() ", .{});
    const allocator = pkgAllocator.allocator;
    var tensor = try Tensor(f64).init(&allocator);
    defer tensor.deinit();
    const size = tensor.getSize();
    try std.testing.expect(size == 0);
    try std.testing.expect(&allocator == tensor.allocator);
}

test "initialization fromShape" {
    std.debug.print("\n     test:initialization fromShape", .{});
    const allocator = pkgAllocator.allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).fromShape(&allocator, &shape);
    defer tensor.deinit();
    const size = tensor.getSize();
    try std.testing.expect(size == 6);
    for (0..tensor.size) |i| {
        const val = try tensor.get(i);
        try std.testing.expect(val == 0);
    }
}

test "Get_Set_Test" {
    std.debug.print("\n     test:Get_Set_Test", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try tensor.set(5, 33);
    const val = try tensor.get(5);

    try std.testing.expect(val == 33);
}

test "Flatten Index Test" {
    std.debug.print("\n     test:Flatten Index Test", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var indices = [_]usize{ 1, 2 };
    const flatIndex = try tensor.flatten_index(&indices);

    //std.debug.print("\nflatIndex: {}\n", .{flatIndex});
    try std.testing.expect(flatIndex == 5);
    indices = [_]usize{ 0, 0 };
    const flatIndex2 = try tensor.flatten_index(&indices);
    //std.debug.print("\nflatIndex2: {}\n", .{flatIndex2});
    try std.testing.expect(flatIndex2 == 0);
}

test "Get_at Set_at Test" {
    std.debug.print("\n     test:Get_at Set_at Test", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var indices = [_]usize{ 1, 1 };
    var value = try tensor.get_at(&indices);
    try std.testing.expect(value == 5.0);

    for (0..2) |i| {
        for (0..3) |j| {
            indices[0] = i;
            indices[1] = j;
            value = try tensor.get_at(&indices);
            try std.testing.expect(value == i * 3 + j + 1);
        }
    }

    try tensor.set_at(&indices, 1.0);
    value = try tensor.get_at(&indices);
    try std.testing.expect(value == 1.0);
}

test "init than fill " {
    std.debug.print("\n     test:init than fill ", .{});
    const allocator = pkgAllocator.allocator;

    var tensor = try Tensor(u8).init(&allocator);
    defer tensor.deinit();

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    try tensor.fill(&inputArray, &shape);

    try std.testing.expect(tensor.data[0] == 1);
    try std.testing.expect(tensor.data[1] == 2);
    try std.testing.expect(tensor.data[2] == 3);
    try std.testing.expect(tensor.data[3] == 4);
    try std.testing.expect(tensor.data[4] == 5);
    try std.testing.expect(tensor.data[5] == 6);
}

test "fromArray than fill " {
    std.debug.print("\n     test:fromArray than fill ", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 10, 20, 30 },
        [_]u8{ 40, 50, 60 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var inputArray2: [3][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
        [_]u8{ 7, 8, 9 },
    };
    var shape2: [2]usize = [_]usize{ 3, 3 };

    try tensor.fill(&inputArray2, &shape2);

    try std.testing.expect(tensor.data[0] == 1);
    try std.testing.expect(tensor.data[1] == 2);
    try std.testing.expect(tensor.data[2] == 3);
    try std.testing.expect(tensor.data[3] == 4);
    try std.testing.expect(tensor.data[4] == 5);
    try std.testing.expect(tensor.data[5] == 6);
    try std.testing.expect(tensor.data[6] == 7);
    try std.testing.expect(tensor.data[7] == 8);
    try std.testing.expect(tensor.data[8] == 9);
}

test " copy() method" {
    std.debug.print("\n     test:copy() method ", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 10, 20, 30 },
        [_]u8{ 40, 50, 60 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var tensorCopy = try tensor.copy();
    defer tensorCopy.deinit();

    for (0..tensor.data.len) |i| {
        try std.testing.expect(tensor.data[i] == tensorCopy.data[i]);
    }

    for (0..tensor.shape.len) |i| {
        try std.testing.expect(tensor.shape[i] == tensorCopy.shape[i]);
    }
}

test "to array " {
    std.debug.print("\n     test:to array ", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();
    const array_from_tensor = try tensor.toArray(shape.len);
    defer allocator.free(array_from_tensor);

    try expect(array_from_tensor.len == 2);
    try expect(array_from_tensor[0].len == 3);
}

test "Reshape" {
    std.debug.print("\n     test: Reshape ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 4 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    const old_size = tensor.size;

    var new_shape: [2]usize = [_]usize{ 3, 2 };

    try tensor.reshape(&new_shape);

    try expect(old_size == tensor.size);
    try expect(tensor.shape[0] == 3);
    try expect(tensor.shape[1] == 2);
}

test "test setToZero() " {
    std.debug.print("\n     test: setToZero()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]u8 = [_][3][3]u8{
        [_][3]u8{
            [_]u8{ 10, 20, 30 },
            [_]u8{ 40, 50, 60 },
            [_]u8{ 70, 80, 90 },
        },
        [_][3]u8{
            [_]u8{ 10, 20, 30 },
            [_]u8{ 40, 50, 60 },
            [_]u8{ 70, 80, 90 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try tensor.setToZero();

    for (tensor.data) |d| {
        try std.testing.expectEqual(d, 0);
    }

    for (0..tensor.shape.len) |i| {
        try std.testing.expectEqual(tensor.shape[i], shape[i]);
    }
}

test "gather along axis 0 and axis 1" {
    const allocator = pkgAllocator.allocator;

    // -------------------------------------------------------------------------------------
    // Test Case 1: Gather Along Axis 0
    // -------------------------------------------------------------------------------------
    std.debug.print("\n     test: gather along axis 0", .{});

    // Initialize input tensor: 3x3 matrix
    var inputArray0: [3][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
        [_]u8{ 7, 8, 9 },
    };
    var inputShape0: [2]usize = [_]usize{ 3, 3 };
    var inputTensor0 = try Tensor(u8).fromArray(&allocator, &inputArray0, &inputShape0);
    defer inputTensor0.deinit();

    // Initialize indices tensor: [0, 2]
    var indicesArray0: [2]usize = [_]usize{ 0, 2 };
    var indicesShape0: [1]usize = [_]usize{2};
    var indicesTensor0 = try Tensor(usize).fromArray(&allocator, &indicesArray0, &indicesShape0);
    defer indicesTensor0.deinit();

    // Perform gather along axis 0
    var gatheredTensor0 = try inputTensor0.gather(indicesTensor0, 0);
    defer gatheredTensor0.deinit();

    // Expected output tensor: [1,2,3,7,8,9], shape [2,3]
    const expectedData0: [6]u8 = [_]u8{ 1, 2, 3, 7, 8, 9 };
    const expectedShape0: [2]usize = [_]usize{ 2, 3 };

    // Check shape
    try std.testing.expect(gatheredTensor0.shape.len == expectedShape0.len);
    for (0..expectedShape0.len) |i| {
        try std.testing.expect(gatheredTensor0.shape[i] == expectedShape0[i]);
    }

    // Check data
    try std.testing.expect(gatheredTensor0.size == 6);
    for (0..gatheredTensor0.size) |i| {
        try std.testing.expect(gatheredTensor0.data[i] == expectedData0[i]);
    }

    // -------------------------------------------------------------------------------------
    // Test Case 2: Gather Along Axis 1
    // -------------------------------------------------------------------------------------
    std.debug.print("\n     test: gather along axis 1", .{});

    var inputArray1: [2][4]u8 = [_][4]u8{
        [_]u8{ 10, 20, 30, 40 },
        [_]u8{ 50, 60, 70, 80 },
    };
    var inputShape1: [2]usize = [_]usize{ 2, 4 };
    var inputTensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &inputShape1);
    defer inputTensor1.deinit();

    var indicesArray1: [2][2]usize = [_][2]usize{
        [_]usize{ 1, 3 },
        [_]usize{ 0, 2 },
    };
    var indicesShape1: [2]usize = [_]usize{ 2, 2 };
    var indicesTensor1 = try Tensor(usize).fromArray(&allocator, &indicesArray1, &indicesShape1);
    defer indicesTensor1.deinit();

    // Perform gather along axis 1
    var gatheredTensor1 = try inputTensor1.gather(indicesTensor1, 1);
    defer gatheredTensor1.deinit();

    // Expected output tensor: [
    //   [20, 40],
    //   [10, 30],
    //   [60, 80],
    //   [50, 70]
    // ], shape [2, 2, 2]
    const expectedData1: [8]u8 = [_]u8{ 20, 40, 10, 30, 60, 80, 50, 70 };
    const expectedShape1: [3]usize = [_]usize{ 2, 2, 2 };

    // Check shape
    try std.testing.expect(gatheredTensor1.shape.len == expectedShape1.len);
    for (0..expectedShape1.len) |i| {
        try std.testing.expect(gatheredTensor1.shape[i] == expectedShape1[i]);
    }

    // Check data
    std.debug.print("\n     gatheredTensor1.size: {}\n", .{gatheredTensor1.size});
    gatheredTensor1.print();

    try std.testing.expect(gatheredTensor1.size == 8);
    for (0..gatheredTensor1.size) |i| {
        std.debug.print("\n     gatheredTensor1.data[i]: {}\n", .{expectedData1[i]});
        std.debug.print("\n     expectedData1[i]: {}\n", .{gatheredTensor1.data[i]});
        try std.testing.expect(gatheredTensor1.data[i] == expectedData1[i]);
    }

    // -------------------------------------------------------------------------------------
    // Test Case 3: Error Handling - Invalid Axis
    // -------------------------------------------------------------------------------------
    std.debug.print("\n     test: gather with invalid axis", .{});
    const invalidAxis: usize = 3; // Input tensor has 2 dimensions
    const result0 = inputTensor0.gather(indicesTensor0, invalidAxis);
    try std.testing.expect(result0 == TensorError.InvalidAxis);
}

test "gather along negative axis" {
    const allocator = pkgAllocator.allocator;

    // -------------------------------------------------------------------------------------
    // Test Case 1: Gather Along Axis 0
    // -------------------------------------------------------------------------------------
    std.debug.print("\n     test: gather along axis 1", .{});

    var inputArray1: [2][4]u8 = [_][4]u8{
        [_]u8{ 10, 20, 30, 40 },
        [_]u8{ 50, 60, 70, 80 },
    };
    var inputShape1: [2]usize = [_]usize{ 2, 4 };
    var inputTensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &inputShape1);
    defer inputTensor1.deinit();

    var indicesArray1: [2][2]usize = [_][2]usize{
        [_]usize{ 1, 3 },
        [_]usize{ 0, 2 },
    };
    var indicesShape1: [2]usize = [_]usize{ 2, 2 };
    var indicesTensor1 = try Tensor(usize).fromArray(&allocator, &indicesArray1, &indicesShape1);
    defer indicesTensor1.deinit();

    // Perform gather along axis 1
    var gatheredTensor1 = try inputTensor1.gather(indicesTensor1, -1);
    defer gatheredTensor1.deinit();

    // Expected output tensor: [
    //   [20, 40],
    //   [10, 30],
    //   [60, 80],
    //   [50, 70]
    // ], shape [2, 2, 2]
    const expectedData1: [8]u8 = [_]u8{ 20, 40, 10, 30, 60, 80, 50, 70 };
    const expectedShape1: [3]usize = [_]usize{ 2, 2, 2 };

    // Check shape
    try std.testing.expect(gatheredTensor1.shape.len == expectedShape1.len);
    for (0..expectedShape1.len) |i| {
        try std.testing.expect(gatheredTensor1.shape[i] == expectedShape1[i]);
    }

    // Check data
    std.debug.print("\n     gatheredTensor1.size: {}\n", .{gatheredTensor1.size});
    gatheredTensor1.print();

    try std.testing.expect(gatheredTensor1.size == 8);
    for (0..gatheredTensor1.size) |i| {
        std.debug.print("\n     gatheredTensor1.data[i]: {}\n", .{expectedData1[i]});
        std.debug.print("\n     expectedData1[i]: {}\n", .{gatheredTensor1.data[i]});
        try std.testing.expect(gatheredTensor1.data[i] == expectedData1[i]);
    }
}

test "slice_onnx basic slicing" {
    std.debug.print("\n     test: slice_onnx basic slicing", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor slicing
    var input_array_1d = [_]i32{ 1, 2, 3, 4, 5 };
    var shape_1d = [_]usize{5};
    var tensor_1d = try Tensor(i32).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Basic slice [1:3]
    var starts = [_]i64{1};
    var ends = [_]i64{3};
    var sliced_1d = try tensor_1d.slice_onnx(&starts, &ends, null, null);
    defer sliced_1d.deinit();

    try std.testing.expectEqual(@as(usize, 2), sliced_1d.size);
    try std.testing.expectEqual(@as(i32, 2), sliced_1d.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced_1d.data[1]);

    // Test 2D tensor slicing
    var input_array_2d = [_][3]i32{
        [_]i32{ 1, 2, 3 },
        [_]i32{ 4, 5, 6 },
        [_]i32{ 7, 8, 9 },
    };
    var shape_2d = [_]usize{ 3, 3 };
    var tensor_2d = try Tensor(i32).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Slice [0:2, 1:3]
    var starts_2d = [_]i64{ 0, 1 };
    var ends_2d = [_]i64{ 2, 3 };
    var sliced_2d = try tensor_2d.slice_onnx(&starts_2d, &ends_2d, null, null);
    defer sliced_2d.deinit();

    try std.testing.expectEqual(@as(usize, 4), sliced_2d.size);
    try std.testing.expectEqual(@as(i32, 2), sliced_2d.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced_2d.data[1]);
    try std.testing.expectEqual(@as(i32, 5), sliced_2d.data[2]);
    try std.testing.expectEqual(@as(i32, 6), sliced_2d.data[3]);
}

test "slice_onnx negative indices" {
    std.debug.print("\n     test: slice_onnx negative indices", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_]i32{ 1, 2, 3, 4, 5 };
    var shape = [_]usize{5};
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test negative indices [-3:-1]
    var starts = [_]i64{-3};
    var ends = [_]i64{-1};
    var sliced = try tensor.slice_onnx(&starts, &ends, null, null);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 2), sliced.size);
    try std.testing.expectEqual(@as(i32, 3), sliced.data[0]);
    try std.testing.expectEqual(@as(i32, 4), sliced.data[1]);
}

test "slice_onnx with steps" {
    std.debug.print("\n     test: slice_onnx with steps", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{6};
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test with step 2
    var starts = [_]i64{0};
    var ends = [_]i64{6};
    var steps = [_]i64{2};
    var sliced = try tensor.slice_onnx(&starts, &ends, null, &steps);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 3), sliced.size);
    try std.testing.expectEqual(@as(i32, 1), sliced.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced.data[1]);
    try std.testing.expectEqual(@as(i32, 5), sliced.data[2]);

    // Test with negative step
    steps[0] = -1;
    starts[0] = 5;
    ends[0] = -1;
    var reversed = try tensor.slice_onnx(&starts, &ends, null, &steps);
    defer reversed.deinit();

    try std.testing.expectEqual(@as(usize, 5), reversed.size);
    try std.testing.expectEqual(@as(i32, 6), reversed.data[0]);
    try std.testing.expectEqual(@as(i32, 5), reversed.data[1]);
    try std.testing.expectEqual(@as(i32, 4), reversed.data[2]);
    try std.testing.expectEqual(@as(i32, 3), reversed.data[3]);
    try std.testing.expectEqual(@as(i32, 2), reversed.data[4]);
}

test "slice_onnx with explicit axes" {
    std.debug.print("\n     test: slice_onnx with explicit axes", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][3]i32{
        [_]i32{ 1, 2, 3 },
        [_]i32{ 4, 5, 6 },
        [_]i32{ 7, 8, 9 },
    };
    var shape = [_]usize{ 3, 3 };
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test slicing only along axis 1
    var starts = [_]i64{1};
    var ends = [_]i64{3};
    var axes = [_]i64{1};
    var sliced = try tensor.slice_onnx(&starts, &ends, &axes, null);
    defer sliced.deinit();

    try std.testing.expectEqual(@as(usize, 2), sliced.shape[1]);
    try std.testing.expectEqual(@as(usize, 3), sliced.shape[0]);
    try std.testing.expectEqual(@as(i32, 2), sliced.data[0]);
    try std.testing.expectEqual(@as(i32, 3), sliced.data[1]);
    try std.testing.expectEqual(@as(i32, 5), sliced.data[2]);
    try std.testing.expectEqual(@as(i32, 6), sliced.data[3]);
    try std.testing.expectEqual(@as(i32, 8), sliced.data[4]);
    try std.testing.expectEqual(@as(i32, 9), sliced.data[5]);
}

test "slice_onnx error cases" {
    std.debug.print("\n     test: slice_onnx error cases", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_]i32{ 1, 2, 3, 4, 5 };
    var shape = [_]usize{5};
    var tensor = try Tensor(i32).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test invalid step size
    var starts = [_]i64{0};
    var ends = [_]i64{5};
    var steps = [_]i64{0}; // Step cannot be 0
    try std.testing.expectError(TensorError.InvalidSliceStep, tensor.slice_onnx(&starts, &ends, null, &steps));

    // Test mismatched lengths
    var starts_2 = [_]i64{ 0, 0 };
    var ends_1 = [_]i64{5};
    try std.testing.expectError(TensorError.InvalidSliceIndices, tensor.slice_onnx(&starts_2, &ends_1, null, null));

    // Test invalid axis
    var axes = [_]i64{5}; // Axis 5 doesn't exist in a 1D tensor
    try std.testing.expectError(TensorError.InvalidSliceIndices, tensor.slice_onnx(&starts, &ends, &axes, null));
}

test "ensure_4D_shape" {
    std.debug.print("\n     test: ensure_4D_shape ", .{});

    //shape 1D
    const shape = [_]usize{5};
    var result = try Tensor(f32).ensure_4D_shape(&shape);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 1);
    try std.testing.expectEqual(result[2], 1);
    try std.testing.expectEqual(result[3], 5);

    //shape 2D
    const shape_2 = [_]usize{ 5, 10 };
    result = try Tensor(f32).ensure_4D_shape(&shape_2);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 1);
    try std.testing.expectEqual(result[2], 5);
    try std.testing.expectEqual(result[3], 10);

    //shape 3D
    std.debug.print("\n     test: ensure_4D_shape with 3 dimensions", .{});

    const shape_3 = [_]usize{ 5, 10, 15 };
    result = try Tensor(f32).ensure_4D_shape(&shape_3);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 1);
    try std.testing.expectEqual(result[1], 5);
    try std.testing.expectEqual(result[2], 10);
    try std.testing.expectEqual(result[3], 15);

    //shape 4D
    std.debug.print("\n     test: ensure_4D_shape with 4 dimensions", .{});

    const shape_4 = [_]usize{ 5, 10, 15, 20 };
    result = try Tensor(f32).ensure_4D_shape(&shape_4);

    try std.testing.expectEqual(result.len, 4);

    try std.testing.expectEqual(result[0], 5);
    try std.testing.expectEqual(result[1], 10);
    try std.testing.expectEqual(result[2], 15);
    try std.testing.expectEqual(result[3], 20);

    // shape 5D --> check for error
    std.debug.print("\n     test: ensure_4D_shape with 5 dimensions", .{});

    const shape_5 = [_]usize{ 5, 10, 15, 20, 25 };
    result = try Tensor(f32).ensure_4D_shape(&shape_5);

    try std.testing.expectError(TensorError.InvalidDimensions, result);
}
