const std = @import("std");
const Tensor = @import("tensor").Tensor;
//import error library
const TensorError = @import("errorHandler").TensorError;
const pkgAllocator = @import("pkgAllocator");

const expect = std.testing.expect;

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

    var tensor_transposed = try tensor.transpose2D();
    defer tensor_transposed.deinit();

    try std.testing.expect(tensor_transposed.data[0] == 1);
    try std.testing.expect(tensor_transposed.data[1] == 4);
    try std.testing.expect(tensor_transposed.data[2] == 2);
    try std.testing.expect(tensor_transposed.data[3] == 5);
    try std.testing.expect(tensor_transposed.data[4] == 3);
    try std.testing.expect(tensor_transposed.data[5] == 6);
}

test "tests isSafe() method" {
    std.debug.print("\n     test: isSafe() method ", .{});

    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try tensor.isSafe();
}

test "tests isSafe() -> TensorError.NotFiniteValue " {
    std.debug.print("\n     test: isSafe()-> TensorError.NotFiniteValue", .{});

    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 8.0, 6.0 },
    };
    const zero: f64 = 1.0;
    inputArray[1][1] = inputArray[1][1] / (zero - 1.0); //NaN here
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensore = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer tensore.deinit();
    try std.testing.expect(std.math.isNan(inputArray[1][1]) == false);
    try std.testing.expect(std.math.isFinite(inputArray[1][1]) == false);
    try std.testing.expectError(TensorError.NotFiniteValue, tensore.isSafe());
}

test "tests isSafe() -> TensorError.NanValue " {
    std.debug.print("\n     test: isSafe()-> TensorError.NanValue", .{});

    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, std.math.nan(f64), 6.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensore = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer tensore.deinit();
    try std.testing.expect(std.math.isNan(inputArray[1][1]) == true);
    try std.testing.expectError(TensorError.NanValue, tensore.isSafe());
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

    try tensor.addPaddingAndDilation(1, 2, 1, 2);

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

    try tensor.addPaddingAndDilation(1, 2, 0, 0);

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

    try tensor.addPaddingAndDilation(0, 0, 1, 2);

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

    var flippedTensor = try tensor.flip();
    defer flippedTensor.deinit();
    //flippedTensor.print();

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
    var resized_1d = try tensor_1d.resize("nearest", &scales, null, "half_pixel");
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
    var resized_2d = try tensor_2d.resize("nearest", &scales_2d, null, "half_pixel");
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
    var resized_1d = try tensor_1d.resize("linear", &scales, null, "half_pixel");
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
    var resized_2d = try tensor_2d.resize("linear", &scales_2d, null, "half_pixel");
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
    var resized_1d = try tensor_1d.resize("cubic", &scales, null, "half_pixel");
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
    var resized = try tensor.resize("nearest", null, &sizes, "half_pixel");
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
        tensor.resize("invalid_mode", &scales, null, "half_pixel"),
    );

    // Test both scales and sizes provided
    var sizes = [_]usize{ 3, 3 };
    try std.testing.expectError(
        TensorError.InvalidInput,
        tensor.resize("nearest", &scales, &sizes, "half_pixel"),
    );

    // Test neither scales nor sizes provided
    try std.testing.expectError(
        TensorError.InvalidInput,
        tensor.resize("nearest", null, null, "half_pixel"),
    );
}
