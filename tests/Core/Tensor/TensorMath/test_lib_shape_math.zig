const std = @import("std");
const pkgAllocator = @import("pkgAllocator");
const TensMath = @import("tensor_m");
const Tensor = @import("tensor").Tensor;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;

test "Concatenate tensors along axis 0" {
    std.debug.print("\n     test: Concatenate tensors along axis 0", .{});
    var allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix for both tensors

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, 0);
    defer result_tensor.deinit();

    const expected_data: [4][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    try std.testing.expect(result_tensor.shape[0] == 4);
    try std.testing.expect(result_tensor.shape[1] == 2);

    for (0..4) |i| {
        for (0..2) |j| {
            //std.debug.print("Checking result_tensor[{d}][{d}]: {f}\n", .{ i, j, result_tensor.data[i * 2 + j] });
            try std.testing.expect(result_tensor.data[i * 2 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate tensors along axis 1" {
    std.debug.print("\n     test: Concatenate tensors along axis 1", .{});
    var allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix for both tensors

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, 1);
    defer result_tensor.deinit();

    const expected_data: [2][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 5.0, 6.0 },
        [_]f32{ 3.0, 4.0, 7.0, 8.0 },
    };
    result_tensor.print();

    try std.testing.expect(result_tensor.shape[0] == 2);
    try std.testing.expect(result_tensor.shape[1] == 4);

    for (0..2) |i| {
        for (0..4) |j| {
            //std.debug.print("Checking result_tensor[{d}][{d}]: {f}\n", .{ i, j, result_tensor.data[i * 4 + j] });
            try std.testing.expect(result_tensor.data[i * 4 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate tensors along negative axis" {
    std.debug.print("\n     test: Concatenate tensors along negative axis", .{});
    var allocator = std.testing.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 6.0 },
        [_]f32{ 7.0, 8.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix for both tensors

    // Initialize tensors from arrays
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var tensors = [_]Tensor(f32){ t1, t2 };

    // Perform concatenation along axis -1 (equivalent to axis 1)
    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, -1);
    defer result_tensor.deinit();

    const expected_data: [2][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 5.0, 6.0 },
        [_]f32{ 3.0, 4.0, 7.0, 8.0 },
    };

    try std.testing.expect(result_tensor.shape[0] == 2);
    try std.testing.expect(result_tensor.shape[1] == 4);

    for (0..2) |i| {
        for (0..4) |j| {
            try std.testing.expect(result_tensor.data[i * 4 + j] == expected_data[i][j]);
        }
    }
}

test "Concatenate 3D tensors along axis 2" {
    std.debug.print("\n     test: Concatenate 3D tensors along axis 2", .{});
    var allocator = std.testing.allocator;

    // Tensor A: shape [2, 2, 2]
    var inputArrayA: [2][2][2]f32 = [_][2][2]f32{
        [_][2]f32{ [_]f32{ 1.0, 2.0 }, [_]f32{ 3.0, 4.0 } },
        [_][2]f32{ [_]f32{ 5.0, 6.0 }, [_]f32{ 7.0, 8.0 } },
    };
    var shapeA: [3]usize = [_]usize{ 2, 2, 2 };
    var tA = try Tensor(f32).fromArray(&allocator, &inputArrayA, &shapeA);
    defer tA.deinit();

    // Tensor B: shape [2, 2, 3]
    var inputArrayB: [2][2][3]f32 = [_][2][3]f32{
        [_][3]f32{ [_]f32{ 9.0, 10.0, 11.0 }, [_]f32{ 12.0, 13.0, 14.0 } },
        [_][3]f32{ [_]f32{ 15.0, 16.0, 17.0 }, [_]f32{ 18.0, 19.0, 20.0 } },
    };
    var shapeB: [3]usize = [_]usize{ 2, 2, 3 };
    var tB = try Tensor(f32).fromArray(&allocator, &inputArrayB, &shapeB);
    defer tB.deinit();

    var tensors = [_]Tensor(f32){ tA, tB };

    // Perform concatenation along axis 2
    var result_tensor = try TensMath.concatenate(f32, &allocator, &tensors, 2);
    defer result_tensor.deinit();

    const expected_data: [2][2][5]f32 = [_][2][5]f32{
        [_][5]f32{
            [_]f32{ 1.0, 2.0, 9.0, 10.0, 11.0 },
            [_]f32{ 3.0, 4.0, 12.0, 13.0, 14.0 },
        },
        [_][5]f32{
            [_]f32{ 5.0, 6.0, 15.0, 16.0, 17.0 },
            [_]f32{ 7.0, 8.0, 18.0, 19.0, 20.0 },
        },
    };

    try std.testing.expect(result_tensor.shape[0] == 2);
    try std.testing.expect(result_tensor.shape[1] == 2);
    try std.testing.expect(result_tensor.shape[2] == 5);

    for (0..2) |i| {
        for (0..2) |j| {
            for (0..5) |k| {
                try std.testing.expect(result_tensor.data[i * 2 * 5 + j * 5 + k] == expected_data[i][j][k]);
            }
        }
    }
}

test "transpose" {
    std.debug.print("\n     test: transpose ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var tensor_transposed = try TensMath.transpose2D(u8, &tensor);
    defer tensor_transposed.deinit();

    try std.testing.expect(tensor_transposed.data[0] == 1);
    try std.testing.expect(tensor_transposed.data[1] == 4);
    try std.testing.expect(tensor_transposed.data[2] == 2);
    try std.testing.expect(tensor_transposed.data[3] == 5);
    try std.testing.expect(tensor_transposed.data[4] == 3);
    try std.testing.expect(tensor_transposed.data[5] == 6);
}

test "transpose multi-dimensions default" {
    std.debug.print("\n     test: transpose multi-dimensions ", .{});
    const allocator = pkgAllocator.allocator;

    // Initialize input Array and shape
    var inputArray: [2][3][4]u8 = [_][3][4]u8{
        [_][4]u8{
            [_]u8{ 1, 2, 3, 4 },
            [_]u8{ 5, 6, 7, 8 },
            [_]u8{ 9, 10, 11, 12 },
        },
        [_][4]u8{
            [_]u8{ 13, 14, 15, 16 },
            [_]u8{ 17, 18, 19, 20 },
            [_]u8{ 21, 22, 23, 24 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 4 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);

    defer tensor.deinit();

    var tensor_transposed = try TensMath.transposeDefault(u8, &tensor);
    defer tensor_transposed.deinit();

    for (0..tensor.size) |i| {
        try std.testing.expect(tensor_transposed.data[i] == tensor.data[i]);
    }
}

test "test addPaddingAndDilation() " {
    std.debug.print("\n     test: addPadding()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try TensMath.addPaddingAndDilation(i8, &tensor, 1, 2, 1, 2);

    var resultArray: [2][7][11]i8 = [_][7][11]i8{
        [_][11]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        },
        [_][11]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 7, 11 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], tensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], tensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, tensor.size);
}

test "test addPaddingAndDilation() -> zero dilatation " {
    std.debug.print("\n     test: addPadding() -> zero dilatation ", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try TensMath.addPaddingAndDilation(i8, &tensor, 1, 2, 0, 0);

    var resultArray: [2][5][7]i8 = [_][5][7]i8{
        [_][7]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 2, 3, 0, 0 },
            [_]i8{ 0, 0, 4, 5, 6, 0, 0 },
            [_]i8{ 0, 0, 7, 8, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
        },
        [_][7]i8{
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 0, 0, 1, 2, 3, 0, 0 },
            [_]i8{ 0, 0, 4, 5, 6, 0, 0 },
            [_]i8{ 0, 0, 7, 8, 9, 0, 0 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 5, 7 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], tensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], tensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, tensor.size);
}

test "test addPaddingAndDilation() -> zero padding" {
    std.debug.print("\n     test: addPaddingAndDilation() -> zero padding", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try TensMath.addPaddingAndDilation(i8, &tensor, 0, 0, 1, 2);

    var resultArray: [2][5][7]i8 = [_][5][7]i8{
        [_][7]i8{
            [_]i8{ 1, 0, 0, 2, 0, 0, 3 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 4, 0, 0, 5, 0, 0, 6 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 7, 0, 0, 8, 0, 0, 9 },
        },
        [_][7]i8{
            [_]i8{ 1, 0, 0, 2, 0, 0, 3 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 4, 0, 0, 5, 0, 0, 6 },
            [_]i8{ 0, 0, 0, 0, 0, 0, 0 },
            [_]i8{ 7, 0, 0, 8, 0, 0, 9 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 5, 7 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();

    tensor.info();
    tensor.print();
    resultTensor.info();
    resultTensor.print();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], tensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], tensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, tensor.size);
}

test "test flip() " {
    std.debug.print("\n     test: flip()", .{});

    const allocator = pkgAllocator.allocator;

    var inputArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 1, 2, 3 },
            [_]i8{ 4, 5, 6 },
            [_]i8{ 7, 8, 9 },
        },
        [_][3]i8{
            [_]i8{ 10, 20, 30 },
            [_]i8{ 40, 50, 60 },
            [_]i8{ 70, 80, 90 },
        },
    };
    var shape: [3]usize = [_]usize{ 2, 3, 3 };
    var tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var flippedTensor = try TensMath.flip(i8, &tensor);
    defer flippedTensor.deinit();
    // DEBUG flippedTensor.print();

    var resultArray: [2][3][3]i8 = [_][3][3]i8{
        [_][3]i8{
            [_]i8{ 9, 8, 7 },
            [_]i8{ 6, 5, 4 },
            [_]i8{ 3, 2, 1 },
        },
        [_][3]i8{
            [_]i8{ 90, 80, 70 },
            [_]i8{ 60, 50, 40 },
            [_]i8{ 30, 20, 10 },
        },
    };

    var resultShape: [3]usize = [_]usize{ 2, 3, 3 };
    var resultTensor = try Tensor(i8).fromArray(&allocator, &resultArray, &resultShape);
    defer resultTensor.deinit();
    std.debug.print("TRY WITH THISSS: \n", .{});
    resultTensor.printMultidim();

    //check on data
    for (0..resultTensor.data.len) |i| {
        try std.testing.expectEqual(resultTensor.data[i], flippedTensor.data[i]);
    }
    //check on shape
    for (0..resultTensor.shape.len) |i| {
        try std.testing.expectEqual(resultTensor.shape[i], flippedTensor.shape[i]);
    }
    //check on size
    try std.testing.expectEqual(resultTensor.size, flippedTensor.size);
}

test "resize with nearest neighbor interpolation" {
    std.debug.print("\n     test: resize with nearest neighbor interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4 };
    var shape_1d = [_]usize{4};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale up by 2x
    var scales = [_]f32{2.0};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "nearest", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 8), resized_1d.size);
    try std.testing.expectEqual(@as(u8, 1), resized_1d.data[0]);
    try std.testing.expectEqual(@as(u8, 1), resized_1d.data[1]);
    try std.testing.expectEqual(@as(u8, 2), resized_1d.data[2]);
    try std.testing.expectEqual(@as(u8, 2), resized_1d.data[3]);

    // Test 2D tensor resize
    var input_array_2d = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape_2d = [_]usize{ 2, 2 };
    var tensor_2d = try Tensor(u8).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Scale up by 2x in both dimensions
    var scales_2d = [_]f32{ 2.0, 2.0 };
    var resized_2d = try TensMath.resize(u8, &tensor_2d, "nearest", &scales_2d, null, "half_pixel");
    defer resized_2d.deinit();

    try std.testing.expectEqual(@as(usize, 16), resized_2d.size);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[1]);
}

test "resize with linear interpolation" {
    std.debug.print("\n     test: resize with linear interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4 };
    var shape_1d = [_]usize{4};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale up by 2x
    var scales = [_]f32{2.0};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "linear", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 8), resized_1d.size);

    // Test 2D tensor resize
    var input_array_2d = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape_2d = [_]usize{ 2, 2 };
    var tensor_2d = try Tensor(u8).fromArray(&allocator, &input_array_2d, &shape_2d);
    defer tensor_2d.deinit();

    // Scale up by 2x in both dimensions
    var scales_2d = [_]f32{ 2.0, 2.0 };
    var resized_2d = try TensMath.resize(u8, &tensor_2d, "linear", &scales_2d, null, "half_pixel");
    defer resized_2d.deinit();

    try std.testing.expectEqual(@as(usize, 16), resized_2d.size);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), resized_2d.shape[1]);
}

test "resize with cubic interpolation" {
    std.debug.print("\n     test: resize with cubic interpolation", .{});
    const allocator = pkgAllocator.allocator;

    // Test 1D tensor resize
    var input_array_1d = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var shape_1d = [_]usize{8};
    var tensor_1d = try Tensor(u8).fromArray(&allocator, &input_array_1d, &shape_1d);
    defer tensor_1d.deinit();

    // Scale down by 0.5x
    var scales = [_]f32{0.5};
    var resized_1d = try TensMath.resize(u8, &tensor_1d, "cubic", &scales, null, "half_pixel");
    defer resized_1d.deinit();

    try std.testing.expectEqual(@as(usize, 4), resized_1d.size);
}

test "resize with explicit sizes" {
    std.debug.print("\n     test: resize with explicit sizes", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Resize to specific dimensions
    var sizes = [_]usize{ 3, 3 };
    var resized = try TensMath.resize(u8, &tensor, "nearest", null, &sizes, "half_pixel");
    defer resized.deinit();

    try std.testing.expectEqual(@as(usize, 9), resized.size);
    try std.testing.expectEqual(@as(usize, 3), resized.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), resized.shape[1]);
}

test "resize error cases" {
    std.debug.print("\n     test: resize error cases", .{});
    const allocator = pkgAllocator.allocator;

    var input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var shape = [_]usize{ 2, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Test invalid mode
    var scales = [_]f32{ 2.0, 2.0 };
    try std.testing.expectError(
        TensorError.UnsupportedMode,
        TensMath.resize(u8, &tensor, "invalid_mode", &scales, null, "half_pixel"),
    );

    // Test both scales and sizes provided
    var sizes = [_]usize{ 3, 3 };
    try std.testing.expectError(
        TensorError.InvalidInput,
        TensMath.resize(u8, &tensor, "nearest", &scales, &sizes, "half_pixel"),
    );

    // Test neither scales nor sizes provided
    try std.testing.expectError(
        TensorError.InvalidInput,
        TensMath.resize(u8, &tensor, "nearest", null, null, "half_pixel"),
    );
}

test "split basic test" {
    std.debug.print("\n     test: split basic test", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x3 tensor
    const input_array = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Split along axis 0 (rows)
    const split_tensors = try TensMath.split(u8, &tensor, 0, null);
    defer {
        for (split_tensors) |*t| {
            t.deinit();
        }
        allocator.free(split_tensors);
    }

    try std.testing.expectEqual(@as(usize, 1), split_tensors.len);
    try std.testing.expectEqual(@as(usize, 2), split_tensors[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 3), split_tensors[0].shape[1]);
    try std.testing.expectEqual(@as(u8, 1), split_tensors[0].data[0]);
    try std.testing.expectEqual(@as(u8, 6), split_tensors[0].data[5]);
}

test "split with custom sizes" {
    std.debug.print("\n     test: split with custom sizes", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 4x2 tensor
    const input_array = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
        [_]u8{ 5, 6 },
        [_]u8{ 7, 8 },
    };
    var shape = [_]usize{ 4, 2 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Split along axis 0 into [1,3] parts
    const split_sizes = [_]usize{ 1, 3 };
    const split_tensors = try TensMath.split(u8, &tensor, 0, &split_sizes);
    defer {
        for (split_tensors) |*t| {
            t.deinit();
        }
        allocator.free(split_tensors);
    }

    try std.testing.expectEqual(@as(usize, 2), split_tensors.len);

    // First split should be 1x2
    try std.testing.expectEqual(@as(usize, 1), split_tensors[0].shape[0]);
    try std.testing.expectEqual(@as(usize, 2), split_tensors[0].shape[1]);
    try std.testing.expectEqual(@as(u8, 1), split_tensors[0].data[0]);
    try std.testing.expectEqual(@as(u8, 2), split_tensors[0].data[1]);

    // Second split should be 3x2
    try std.testing.expectEqual(@as(usize, 3), split_tensors[1].shape[0]);
    try std.testing.expectEqual(@as(usize, 2), split_tensors[1].shape[1]);
    try std.testing.expectEqual(@as(u8, 3), split_tensors[1].data[0]);
    try std.testing.expectEqual(@as(u8, 8), split_tensors[1].data[5]);
}

test "split with negative axis" {
    std.debug.print("\n     test: split with negative axis", .{});
    const allocator = pkgAllocator.allocator;

    // Create a 2x4 tensor
    const input_array = [_][4]u8{
        [_]u8{ 1, 2, 3, 4 },
        [_]u8{ 5, 6, 7, 8 },
    };
    var shape = [_]usize{ 2, 4 };
    var tensor = try Tensor(u8).fromArray(&allocator, &input_array, &shape);
    defer tensor.deinit();

    // Split along axis -1 (last axis) into [2,2] parts
    const split_sizes = [_]usize{ 2, 2 };
    const split_tensors = try TensMath.split(u8, &tensor, -1, &split_sizes);
    defer {
        for (split_tensors) |*t| {
            t.deinit();
        }
        allocator.free(split_tensors);
    }

    try std.testing.expectEqual(@as(usize, 2), split_tensors.len);

    // Both splits should be 2x2
    for (split_tensors) |t| {
        try std.testing.expectEqual(@as(usize, 2), t.shape[0]);
        try std.testing.expectEqual(@as(usize, 2), t.shape[1]);
    }

    // Check first split
    try std.testing.expectEqual(@as(u8, 1), split_tensors[0].data[0]);
    try std.testing.expectEqual(@as(u8, 2), split_tensors[0].data[1]);
    try std.testing.expectEqual(@as(u8, 5), split_tensors[0].data[2]);
    try std.testing.expectEqual(@as(u8, 6), split_tensors[0].data[3]);

    // Check second split
    try std.testing.expectEqual(@as(u8, 3), split_tensors[1].data[0]);
    try std.testing.expectEqual(@as(u8, 4), split_tensors[1].data[1]);
    try std.testing.expectEqual(@as(u8, 7), split_tensors[1].data[2]);
    try std.testing.expectEqual(@as(u8, 8), split_tensors[1].data[3]);
}

test "get_resize_output_shape()" {
    std.debug.print("\n     test: get_resize_output_shape \n", .{});

    var input_shape = [_]usize{ 2, 3, 4 };
    var scales = [_]f32{ 2.0, 1.5, 0.5 };
    var target_sizes = [_]usize{ 4, 4, 2 };

    // Test with scales
    {
        const output_shape = try TensMath.get_resize_output_shape(&input_shape, &scales, null);
        defer pkgAllocator.allocator.free(output_shape);
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }

    // Test with target sizes
    {
        const output_shape = try TensMath.get_resize_output_shape(&input_shape, null, &target_sizes);
        defer pkgAllocator.allocator.free(output_shape);
        try std.testing.expectEqual(@as(usize, 4), output_shape[0]);
        try std.testing.expectEqual(@as(usize, 4), output_shape[1]);
        try std.testing.expectEqual(@as(usize, 2), output_shape[2]);
    }

    // Test invalid input (both scales and sizes null)
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, null, null));

    // Test invalid input (both scales and sizes provided)
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, &scales, &target_sizes));

    // Test mismatched dimensions
    var wrong_scales = [_]f32{ 2.0, 1.5 };
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, &wrong_scales, null));

    var wrong_sizes = [_]usize{ 4, 4 };
    try std.testing.expectError(TensorError.InvalidInput, TensMath.get_resize_output_shape(&input_shape, null, &wrong_sizes));
}

test "get_concatenate_output_shape" {
    std.debug.print("\n     test: get_concatenate_output_shape \n", .{});

    const allocator = pkgAllocator.allocator;

    // Test concatenation along axis 0
    var shapes = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 3, 3 },
        &[_]usize{ 1, 3 },
    };

    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes, 0);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 6), output_shape[0]); // 2 + 3 + 1
        try std.testing.expectEqual(@as(usize, 3), output_shape[1]); // unchanged
    }

    // Test concatenation along axis 1
    const shapes_axis1 = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 2, 2 },
    };

    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes_axis1, 1);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // unchanged
        try std.testing.expectEqual(@as(usize, 5), output_shape[1]); // 3 + 2
    }

    // Test concatenation along negative axis
    {
        const output_shape = try TensMath.get_concatenate_output_shape(&shapes_axis1, -1);
        defer allocator.free(output_shape);

        try std.testing.expectEqual(@as(usize, 2), output_shape.len);
        try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // unchanged
        try std.testing.expectEqual(@as(usize, 5), output_shape[1]); // 3 + 2
    }

    // Test error cases
    var empty_shapes = [_][]const usize{};
    try std.testing.expectError(TensorMathError.EmptyTensorList, TensMath.get_concatenate_output_shape(&empty_shapes, 0));

    var mismatched_shapes = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 3, 4 }, // different non-concat dimension
    };
    try std.testing.expectError(TensorError.MismatchedShape, TensMath.get_concatenate_output_shape(&mismatched_shapes, 0));

    var mismatched_rank_shapes = [_][]const usize{
        &[_]usize{ 2, 3 },
        &[_]usize{ 2, 3, 4 }, // different rank
    };
    try std.testing.expectError(TensorError.MismatchedRank, TensMath.get_concatenate_output_shape(&mismatched_rank_shapes, 0));

    try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.get_concatenate_output_shape(&shapes_axis1, 2));
    try std.testing.expectError(TensorError.AxisOutOfBounds, TensMath.get_concatenate_output_shape(&shapes_axis1, -3));
}

test "get_split_output_shapes()" {
    std.debug.print("\n     test: get_split_output_shapes \n", .{});

    const allocator = pkgAllocator.allocator;
    var input_shape = [_]usize{ 2, 3, 4 };

    // Test with null split_sizes (equal splits)
    {
        const output_shapes = try TensMath.get_split_output_shapes(&input_shape, 1, null);
        defer {
            for (output_shapes) |shape| {
                allocator.free(shape);
            }
            allocator.free(output_shapes);
        }

        try std.testing.expectEqual(@as(usize, 1), output_shapes.len);
        try std.testing.expectEqual(@as(usize, 3), output_shapes[0].len);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[0][0]);
        try std.testing.expectEqual(@as(usize, 3), output_shapes[0][1]);
        try std.testing.expectEqual(@as(usize, 4), output_shapes[0][2]);
    }

    // Test with specific split_sizes
    {
        var split_sizes = [_]usize{ 1, 2 };
        const output_shapes = try TensMath.get_split_output_shapes(&input_shape, 1, &split_sizes);
        defer {
            for (output_shapes) |shape| {
                allocator.free(shape);
            }
            allocator.free(output_shapes);
        }

        try std.testing.expectEqual(@as(usize, 2), output_shapes.len);
        try std.testing.expectEqual(@as(usize, 3), output_shapes[0].len);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[0][0]);
        try std.testing.expectEqual(@as(usize, 1), output_shapes[0][1]);
        try std.testing.expectEqual(@as(usize, 4), output_shapes[0][2]);
        try std.testing.expectEqual(@as(usize, 2), output_shapes[1][1]);
    }

    // Test invalid axis
    try std.testing.expectError(TensorError.InvalidAxis, TensMath.get_split_output_shapes(&input_shape, -4, null));
    try std.testing.expectError(TensorError.InvalidAxis, TensMath.get_split_output_shapes(&input_shape, 3, null));

    // Test invalid split sizes
    var invalid_split_sizes = [_]usize{ 1, 1 };
    try std.testing.expectError(TensorError.InvalidSplitSize, TensMath.get_split_output_shapes(&input_shape, 1, &invalid_split_sizes));
}

test "Empty tensor list error" {
    std.debug.print("\n     test: Empty tensor list error", .{});
    const empty_shapes: []const []const usize = &[_][]const usize{};
    try std.testing.expectError(TensorMathError.EmptyTensorList, TensMath.get_concatenate_output_shape(empty_shapes, 0));
}
