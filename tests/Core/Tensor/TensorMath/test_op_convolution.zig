const std = @import("std");
const pkgAllocator = @import("pkgAllocator");
const TensMath = @import("tensor_m");
const Tensor = @import("tensor").Tensor;
const TensorMathError = @import("errorHandler").TensorMathError;

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

test "get_convolution_output_shape()" {
    std.debug.print("\n     test: get_convolution_output_shape \n", .{});

    var input_shape = [_]usize{ 2, 2, 5, 5 }; // batch=2, channels=2, height=5, width=5
    var kernel_shape = [_]usize{ 3, 2, 3, 3 }; // filters=3, channels=2, height=3, width=3
    var stride = [_]usize{ 1, 1 };

    var output_shape = try TensMath.get_convolution_output_shape(&input_shape, &kernel_shape, &stride);

    try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // batch size
    try std.testing.expectEqual(@as(usize, 3), output_shape[1]); // num filters
    try std.testing.expectEqual(@as(usize, 3), output_shape[2]); // output height
    try std.testing.expectEqual(@as(usize, 3), output_shape[3]); // output width

    // Test with different stride
    stride = [_]usize{ 2, 2 };
    output_shape = try TensMath.get_convolution_output_shape(&input_shape, &kernel_shape, &stride);

    try std.testing.expectEqual(@as(usize, 2), output_shape[0]); // batch size
    try std.testing.expectEqual(@as(usize, 3), output_shape[1]); // num filters
    try std.testing.expectEqual(@as(usize, 2), output_shape[2]); // output height
    try std.testing.expectEqual(@as(usize, 2), output_shape[3]); // output width

    // Test invalid dimensions
    var invalid_input_shape = [_]usize{ 2, 2, 5 };
    try std.testing.expectError(TensorMathError.InvalidDimensions, TensMath.get_convolution_output_shape(&invalid_input_shape, &kernel_shape, &stride));

    // Test invalid stride
    var invalid_stride = [_]usize{ 0, 1 };
    try std.testing.expectError(TensorMathError.WrongStride, TensMath.get_convolution_output_shape(&input_shape, &kernel_shape, &invalid_stride));
}
