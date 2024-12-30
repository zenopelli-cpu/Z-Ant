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
