const std = @import("std");
const Tensor = @import("tensor").Tensor;
const TensMath = @import("tensor_m");
const TensorMathError = @import("errorHandler").TensorMathError;
const TensorError = @import("errorHandler").TensorError;
const ArchitectureError = @import("errorHandler").ArchitectureError;
const ErrorHandler = @import("errorHandler");
const PoolingType = @import("poolingLayer").PoolingType;
const pkgAllocator = @import("pkgAllocator");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test "Sum two tensors on CPU architecture" {
    std.debug.print("\n     test: Sum two tensors on CPU architecture", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2.deinit();

    var t3 = try TensMath.sum_tensors(f32, f64, &t1, &t2); // Output tensor with larger type
    defer t3.deinit();

    // Check if the values in t3 are as expected
    try std.testing.expect(2.0 == t3.data[0]);
    try std.testing.expect(4.0 == t3.data[1]);
}

test " equal() " {
    std.debug.print("\n     test:equal()", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2.deinit();

    try std.testing.expect(TensMath.equal(f32, &t1, &t2) == true);

    //wrong data
    var inputArray3: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 333.0 },
    };

    var shape3: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t3 = try Tensor(f32).fromArray(&allocator, &inputArray3, &shape3);
    defer t3.deinit();

    try std.testing.expect(TensMath.equal(f32, &t3, &t2) == false);

    //wrong shape

    var shape4: [2]usize = [_]usize{ 1, 4 }; // 2x2 matrix

    var t4 = try Tensor(f32).fromArray(&allocator, &inputArray3, &shape4);
    defer t4.deinit();

    try std.testing.expect(TensMath.equal(f32, &t4, &t2) == false);
}

test "Error when input tensors have different sizes" {
    std.debug.print("\n     test: Error when input tensors have different sizes", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
        [_]f32{ 14.0, 15.0 },
    };

    var shape1 = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2 = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorDifferentSize, TensMath.sum_tensors(f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "test tensor element-wise multiplication" {
    std.debug.print("\n     test: tensor element-wise multiplication ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray1: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &shape);
    defer tensor1.deinit();

    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();

    var tensor3 = try TensMath.mul(u8, &tensor1, &tensor2);
    defer tensor3.deinit();

    for (0..tensor3.size) |i| {
        try std.testing.expect(tensor3.data[i] == tensor1.data[i] * tensor2.data[i]);
    }
}

test "Dot product 2x2" {
    std.debug.print("\n     test:Dot product 2x2", .{});

    const allocator = pkgAllocator.allocator;

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);

    var result_tensor = try TensMath.dot_product_tensor(f32, f64, &t1, &t2);

    try std.testing.expect(9.0 == result_tensor.data[0]);
    try std.testing.expect(12.0 == result_tensor.data[1]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible sizes for dot product" {
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(f32, f64, &t1, &t2));

    _ = TensMath.dot_product_tensor(f32, f64, &t1, &t2) catch |err| {
        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible shapes for dot product" {
    std.debug.print("\n     test: Error when input tensors have incompatible shapes for dot product", .{});
    const allocator = pkgAllocator.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 4x1 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "add bias" {
    std.debug.print("\n     test:add bias", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [2]usize = [_]usize{ 2, 3 }; // 2x3 matrix
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    const flatArr: [6]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var shape_bias: [1]usize = [_]usize{3};
    var bias_array: [3]f32 = [_]f32{ 1.0, 1.0, 1.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &shape_bias);

    try TensMath.add_bias(f32, &t1, &bias);

    for (t1.data, 0..) |*data, i| {
        try std.testing.expect(data.* == flatArr[i] + 1);
    }

    t1.deinit();
    bias.deinit();
}

test "mean" {
    std.debug.print("\n     test:mean", .{});
    const allocator = pkgAllocator.allocator;

    var shape_tensor: [1]usize = [_]usize{3}; // 2x3 matrix
    var inputArray: [3]f32 = [_]f32{ 1.0, 2.0, 3.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);

    try std.testing.expect(2.0 == TensMath.mean(f32, &t1));

    t1.deinit();
}

test "Convolution 4D Input with 2x2x2x2 Kernel shape" {
    std.debug.print("\n     test: Convolution 4D Input with 2x2x2x2 Kernel shape\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
        [_][3][3]f32{
            // First Channel
            [_][3]f32{
                [_]f32{ 2.0, 2.0, 3.0 },
                [_]f32{ 4.0, 5.0, 6.0 },
                [_]f32{ 7.0, 8.0, 9.0 },
            },
            // Second Channel
            [_][3]f32{
                [_]f32{ 8.0, 8.0, 7.0 },
                [_]f32{ 6.0, 5.0, 4.0 },
                [_]f32{ 3.0, 2.0, 1.0 },
            },
        },
        // Second batch
        [_][3][3]f32{
            // First channel
            [_][3]f32{
                [_]f32{ 2.0, 3.0, 4.0 },
                [_]f32{ 5.0, 6.0, 7.0 },
                [_]f32{ 8.0, 9.0, 10.0 },
            },
            // Second channel
            [_][3]f32{
                [_]f32{ 10.0, 9.0, 8.0 },
                [_]f32{ 7.0, 6.0, 5.0 },
                [_]f32{ 4.0, 3.0, 2.0 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };

    var inputbias: [2]f32 = [_]f32{ 1, 1 }; //batches: 2, filters:2

    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(f32).fromArray(&allocator, &inputbias, &bias_shape);
    defer bias.deinit();
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    const stride: [2]usize = [_]usize{ 1, 1 };

    var result_tensor = try TensMath.convolve_tensor_with_bias(f32, &input_tensor, &kernel_tensor, &bias, &stride);
    defer result_tensor.deinit();

    // Expected results with the correct dimensions
    const expected_result: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 3.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 5.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
    };

    // result_tensor.info();
    // result_tensor.print();

    const output_location = try allocator.alloc(usize, 4); //coordinates in the output space, see test below
    defer allocator.free(output_location);
    @memset(output_location, 0);

    for (0..2) |batch| {
        output_location[0] = batch;
        for (0..2) |filter| {
            output_location[1] = filter;
            for (0..2) |row| {
                output_location[2] = row;
                for (0..2) |col| {
                    output_location[3] = col;
                    //std.debug.print("\n get OUTPUT at:{any}", .{output_location});
                    try std.testing.expectEqual(expected_result[batch][filter][row][col], result_tensor.get_at(output_location));
                }
            }
        }
    }
}

test "convolution_backward_biases() " {
    std.debug.print("\n     test: convolution_backward_biases \n", .{});

    const allocator = pkgAllocator.allocator;

    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var d_val_array: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 3.0, 5.0 },
                [_]f32{ 5.0, 5.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 14.0, 14.0 },
                [_]f32{ 14.0, 14.0 },
            },
            [_][2]f32{
                [_]f32{ 10.0, 10.0 },
                [_]f32{ 10.0, 10.0 },
            },
        },
    };

    var d_val = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val.deinit();

    //compute bias derivate
    var result_tensor = try TensMath.convolution_backward_biases(f32, &d_val);
    defer result_tensor.deinit();

    result_tensor.print();

    var d_bias_shape: [1]usize = [_]usize{2};
    var d_bias_array: [2]f32 = [_]f32{ 74, 44 };

    var d_bias_expected = try Tensor(f32).fromArray(&allocator, &d_bias_array, &d_bias_shape);
    defer d_bias_expected.deinit();

    //check on values
    for (d_bias_expected.data, 0..) |expected, i| {
        try std.testing.expectEqual(expected, result_tensor.data[i]);
    }

    //check on dim
    try std.testing.expectEqual(d_bias_expected.size, result_tensor.size);

    for (d_bias_expected.shape, 0..) |expected, i| {
        try std.testing.expectEqual(expected, result_tensor.shape[i]);
    }
}

test "convolution_backward_weights() " {
    std.debug.print("\n     test: convolution_backward_weights \n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 4 };
    var inputArray: [2][2][3][4]f32 = [_][2][3][4]f32{ //batches:2, channels:2, rows:3, cols:4
        //First Batch
        [_][3][4]f32{
            // First Channel
            [_][4]f32{
                [_]f32{ 2.0, 2.0, 3.0, 0.0 },
                [_]f32{ 4.0, 5.0, 6.0, 0.0 },
                [_]f32{ 7.0, 8.0, 9.0, 0.0 },
            },
            // Second Channel
            [_][4]f32{
                [_]f32{ 8.0, 8.0, 7.0, 0.0 },
                [_]f32{ 6.0, 5.0, 4.0, 0.0 },
                [_]f32{ 3.0, 2.0, 1.0, 0.0 },
            },
        },
        // Second batch
        [_][3][4]f32{
            // First channel
            [_][4]f32{
                [_]f32{ 2.0, 3.0, 4.0, 0.0 },
                [_]f32{ 5.0, 6.0, 7.0, 0.0 },
                [_]f32{ 8.0, 9.0, 10.0, 0.0 },
            },
            // Second channel
            [_][4]f32{
                [_]f32{ 10.0, 9.0, 8.0, 0.0 },
                [_]f32{ 7.0, 6.0, 5.0, 0.0 },
                [_]f32{ 4.0, 3.0, 2.0, 0.0 },
            },
        },
    };
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    // d_val
    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 3 };
    var d_val_array: [2][2][2][3]f32 = [_][2][2][3]f32{
        // Primo batch
        [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 0, 0, 0 },
                [_]f32{ 0, 0, 0 },
            },
            [_][3]f32{
                [_]f32{ 0, 0, 0 },
                [_]f32{ 0, 0, 0 },
            },
        },
        // Secondo batch
        [_][2][3]f32{
            [_][3]f32{
                [_]f32{ 1.0, 1.0, 1.0 },
                [_]f32{ 1.0, 1.0, 1.0 },
            },
            [_][3]f32{
                [_]f32{ 1.0, 1.0, 1.0 },
                [_]f32{ 1.0, 1.0, 1.0 },
            },
        },
    };

    var d_val_tensor = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val_tensor.deinit();

    //create stride
    const stride: [2]usize = [_]usize{ 1, 1 };

    //creating all zero bias
    var bias_array: [2]f32 = [_]f32{ 1, 1 }; //batches: 2, filters:2
    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &bias_shape);
    defer bias.deinit();

    //generating an output
    var output_tensor = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer output_tensor.deinit();

    output_tensor.info();
    output_tensor.print();

    //compute bias derivate
    var d_weights = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer d_weights.deinit();
}

test "convolution_backward_weights() small" {
    std.debug.print("\n     test: convolution_backward_weights \n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
        [_][3][3]f32{
            // First Channel
            [_][3]f32{
                [_]f32{ 2.0, 2.0, 3.0 },
                [_]f32{ 4.0, 5.0, 6.0 },
                [_]f32{ 7.0, 8.0, 9.0 },
            },
            // Second Channel
            [_][3]f32{
                [_]f32{ 8.0, 8.0, 7.0 },
                [_]f32{ 6.0, 5.0, 4.0 },
                [_]f32{ 3.0, 2.0, 1.0 },
            },
        },
        // Second batch
        [_][3][3]f32{
            // First channel
            [_][3]f32{
                [_]f32{ 2.0, 3.0, 4.0 },
                [_]f32{ 5.0, 6.0, 7.0 },
                [_]f32{ 8.0, 9.0, 10.0 },
            },
            // Second channel
            [_][3]f32{
                [_]f32{ 10.0, 9.0, 8.0 },
                [_]f32{ 7.0, 6.0, 5.0 },
                [_]f32{ 4.0, 3.0, 2.0 },
            },
        },
    };
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    // d_val
    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var d_val_array: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
    };

    var d_val_tensor = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val_tensor.deinit();

    //create stride
    const stride: [2]usize = [_]usize{ 1, 1 };

    //creating all zero bias
    var bias_array: [2]f32 = [_]f32{ 1, 1 }; //batches: 2, filters:2
    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &bias_shape);
    defer bias.deinit();

    //generating an output
    var output_tensor = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer output_tensor.deinit();

    output_tensor.info();
    output_tensor.print();

    //compute bias derivate
    var d_weights = try TensMath.convolution_backward_weights(
        f32,
        &input_tensor,
        &d_val_tensor,
        kernel_tensor.shape[0..],
        stride,
    );
    defer d_weights.deinit();

    // d_weights.info();
    // d_weights.print();
}

test "convolution_backward_input() " {
    std.debug.print("\n     test: convolution_backward_input() \n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
        [_][3][3]f32{
            // First Channel
            [_][3]f32{
                [_]f32{ 2.0, 2.0, 3.0 },
                [_]f32{ 4.0, 5.0, 6.0 },
                [_]f32{ 7.0, 8.0, 9.0 },
            },
            // Second Channel
            [_][3]f32{
                [_]f32{ 8.0, 8.0, 7.0 },
                [_]f32{ 6.0, 5.0, 4.0 },
                [_]f32{ 3.0, 2.0, 1.0 },
            },
        },
        // Second batch
        [_][3][3]f32{
            // First channel
            [_][3]f32{
                [_]f32{ 2.0, 3.0, 4.0 },
                [_]f32{ 5.0, 6.0, 7.0 },
                [_]f32{ 8.0, 9.0, 10.0 },
            },
            // Second channel
            [_][3]f32{
                [_]f32{ 10.0, 9.0, 8.0 },
                [_]f32{ 7.0, 6.0, 5.0 },
                [_]f32{ 4.0, 3.0, 2.0 },
            },
        },
    };
    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]f32 = [_][2][2][2]f32{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
        //second filter
        [_][2][2]f32{
            //first channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
            //second channel
            [_][2]f32{
                [_]f32{ 0.0, 0.0 },
                [_]f32{ 0.0, 0.0 },
            },
        },
    };
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    // d_val
    var d_val_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var d_val_array: [2][2][2][2]f32 = [_][2][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, 1.0 },
                [_]f32{ 1.0, 1.0 },
            },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
            [_][2]f32{
                [_]f32{ 0, 0 },
                [_]f32{ 0, 0 },
            },
        },
    };

    var d_val_tensor = try Tensor(f32).fromArray(&allocator, &d_val_array, &d_val_shape);
    defer d_val_tensor.deinit();

    //create stride
    const stride: [2]usize = [_]usize{ 1, 1 };

    //creating all zero bias
    var bias_array: [2][2]f32 = [_][2]f32{ //batches: 2, filters:2
        [_]f32{ 1, 1 },
        [_]f32{ 10, 10 },
    };
    var bias_shape: [2]usize = [_]usize{ 2, 2 };
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &bias_shape);
    defer bias.deinit();

    //generating an output
    var output_tensor = try TensMath.convolution_backward_input(
        f32,
        &d_val_tensor,
        &kernel_tensor,
        input_tensor.shape[0..],
        stride,
    );
    defer output_tensor.deinit();

    output_tensor.info();
    output_tensor.print();

    //compute bias derivate
    var d_input = try TensMath.convolution_backward_input(
        f32,
        &d_val_tensor,
        &kernel_tensor,
        input_tensor.shape[0..],
        stride,
    );
    defer d_input.deinit();

    std.debug.print("\n   ----------------------------------------------  \n", .{});
    d_input.print();
    d_input.info();
}

test "Pooling 2D" {
    std.debug.print("\n     test: Pooling 2D\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3 input, same as original
    var shape_tensor: [2]usize = [_]usize{ 3, 3 };
    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 40.0, 50.0, 60.0 },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    // Calculate W = number of windows
    const input_rows = input1.shape[0];
    const input_cols = input1.shape[1];
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; // 2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; // 2
    const W = out_rows * out_cols; // 4

    // Instead of used_input being shape of input, now [W,3,3]
    var used_input1_shape = [_]usize{ W, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output1: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output1.deinit();

    // Same checks for output as original
    try std.testing.expectEqual(output1.shape.len, input1.shape.len);
    try std.testing.expectEqual(output1.shape[0], 2);
    try std.testing.expectEqual(output1.shape[1], 2);

    try std.testing.expectEqual(output1.data[0], 5);
    try std.testing.expectEqual(output1.data[1], 6);
    try std.testing.expectEqual(output1.data[2], 50);
    try std.testing.expectEqual(output1.data[3], 60);

    // Now we have 4 windows (W=4), each 3x3
    // The original code expected certain pattern in used_input:
    // (1,1)=5 for first window
    // (1,2)=6 for second window
    // (2,1)=50 for third window
    // (2,2)=60 for fourth window
    //
    // Windows:
    // w=0: top-left max at (1,1)
    // w=1: top-right max at (1,2)
    // w=2: bottom-left max at (2,1)
    // w=3: bottom-right max at (2,2)

    // Check w=0 window:
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[0 * 9 + 1 * 3 + 1]);

    // w=1: top-right (1,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[1 * 9 + 1 * 3 + 2]);

    // w=2: bottom-left (2,1)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[2 * 9 + 2 * 3 + 1]);

    // w=3: bottom-right (2,2)
    try std.testing.expectEqual(@as(u1, 1), used_input1.data[3 * 9 + 2 * 3 + 2]);

    // Another test with kernel2 and stride2
    var kernel2: [2]usize = [2]usize{ 2, 2 };
    var stride2: [2]usize = [2]usize{ 2, 2 };

    const out_rows_2 = (3 - 2 + 1) / 2;
    const out_cols_2 = (3 - 2 + 1) / 2;
    const W2 = out_rows_2 * out_cols_2;

    var used_input2_shape = [_]usize{ W2, 3, 3 };
    var used_input2 = try Tensor(u8).fromShape(&allocator, used_input2_shape[0..]);
    defer used_input2.deinit();
    for (used_input2.data) |*val| val.* = 0;

    var output2: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input2, &kernel2, &stride2, PoolingType.Max);
    defer output2.deinit();

    try std.testing.expectEqual(output2.shape.len, input1.shape.len);
    try std.testing.expectEqual(output2.shape[0], 1);
    try std.testing.expectEqual(output2.shape[1], 1);
    try std.testing.expectEqual(output2.data[0], 5);

    // single window w=0 max at (1,1)
    try std.testing.expectEqual(@as(u1, 1), used_input2.data[0 * (3 * 3) + 1 * 3 + 1]);
}

test "Pooling multidim" {
    std.debug.print("\n     test: Pooling multidim\n", .{});

    const allocator = pkgAllocator.allocator;

    // 3x3x3 input as original
    var shape_tensor: [3]usize = [_]usize{ 3, 3, 3 };
    var inputArray: [3][3][3]f32 = [_][3][3]f32{
        [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ 10.0, 20.0, 30.0 },
            [_]f32{ 40.0, 0.0, -10.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ -1.0, -2.0, -3.0 },
            [_]f32{ -4.0, -5.0, -6.0 },
            [_]f32{ -40.0, -50.0, -60.0 },
        },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    const d = input1.shape[0]; // 3
    const input_rows = input1.shape[1]; //3
    const input_cols = input1.shape[2]; //3
    const out_rows = (input_rows - kernel1[0] + 1) / stride1[0]; //2
    const out_cols = (input_cols - kernel1[1] + 1) / stride1[1]; //2
    const W_tot = d * out_rows * out_cols; // 3*2*2=12

    var used_input1_shape = [_]usize{ W_tot, input_rows, input_cols };
    var used_input1 = try Tensor(u8).fromShape(&allocator, used_input1_shape[0..]);
    defer used_input1.deinit();
    for (used_input1.data) |*val| val.* = 0;

    var output: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output.deinit();

    try std.testing.expectEqual(output.shape.len, input1.shape.len);
    try std.testing.expectEqual(output.shape[0], 3);
    try std.testing.expectEqual(output.shape[1], 2);
    try std.testing.expectEqual(output.shape[2], 2);

    // We do not check used_input details here, just shapes as in original.
}

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

test "Subtraction with same shape tensors" {
    std.debug.print("\n     test: Subtraction with same shape tensors", .{});
    const allocator = pkgAllocator.allocator;

    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };

    var inputArray2: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 4.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 6.0);
    try std.testing.expectEqual(result.data[3], 7.0);
}

test "Subtraction with broadcasting - scalar and matrix" {
    std.debug.print("\n     test: Subtraction with broadcasting - scalar and matrix", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Scalar (1x1)
    var inputArray2 = [_]f32{2.0};
    var shape2: [1]usize = [_]usize{1};

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 3.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 7.0);
    try std.testing.expectEqual(result.data[3], 9.0);
}

test "Subtraction with broadcasting - row and matrix" {
    std.debug.print("\n     test: Subtraction with broadcasting - row and matrix", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Row vector as 2D array with broadcasting shape
    var inputArray2: [1][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
    };
    var shape2: [2]usize = [_]usize{ 1, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    var result = try TensMath.sub_tensors(f32, f32, &t1, &t2);
    defer result.deinit();

    try std.testing.expectEqual(result.data[0], 4.0);
    try std.testing.expectEqual(result.data[1], 5.0);
    try std.testing.expectEqual(result.data[2], 8.0);
    try std.testing.expectEqual(result.data[3], 9.0);
}

test "Subtraction with incompatible shapes" {
    std.debug.print("\n     test: Subtraction with incompatible shapes", .{});
    const allocator = pkgAllocator.allocator;

    // Matrix 2x2
    var inputArray1: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 7.0 },
        [_]f32{ 9.0, 11.0 },
    };
    var shape1: [2]usize = [_]usize{ 2, 2 };

    // Matrix 3x2 (incompatible)
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 3.0, 4.0 },
        [_]f32{ 5.0, 6.0 },
    };
    var shape2: [2]usize = [_]usize{ 3, 2 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray1, &shape1);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);
    defer t2.deinit();

    try std.testing.expectError(TensorMathError.IncompatibleBroadcastShapes, TensMath.sub_tensors(f32, f32, &t1, &t2));
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

    try TensMath.isSafe(u8, &tensor);
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
    try std.testing.expectError(TensorError.NotFiniteValue, TensMath.isSafe(f64, &tensore));
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
    try std.testing.expectError(TensorError.NanValue, TensMath.isSafe(f64, &tensore));
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

test "test tensor element-wise divisio" {
    std.debug.print("\n     test: tensor element-wise division ", .{});
    const allocator = pkgAllocator.allocator;

    // Inizializzazione degli array di input
    var inputArray1: [2][3]u8 = [_][3]u8{
        [_]u8{ 2, 4, 6 },
        [_]u8{ 8, 10, 12 },
    };

    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &shape);
    defer tensor1.deinit();

    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();

    var tensor3 = try TensMath.div(u8, &tensor1, &tensor2);
    defer tensor3.deinit();

    for (0..tensor3.size) |i| {
        try std.testing.expect(tensor3.data[i] == tensor1.data[i] / tensor2.data[i]);
    }
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
