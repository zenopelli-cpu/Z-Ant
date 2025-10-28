const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.test_lib_shape);

// test "gather along axis 0 and axis 1" {
//     const allocator = pkgAllocator.allocator;

//     // -------------------------------------------------------------------------------------
//     // Test Case 1: Gather Along Axis 0
//     // -------------------------------------------------------------------------------------
//     tests_log.info("\n     test: gather along axis 0", .{});

//     // Initialize input tensor: 3x3 matrix
//     var inputArray0: [3][3]u8 = [_][3]u8{
//         [_]u8{ 1, 2, 3 },
//         [_]u8{ 4, 5, 6 },
//         [_]u8{ 7, 8, 9 },
//     };
//     var inputShape0: [2]usize = [_]usize{ 3, 3 };
//     var inputTensor0 = try Tensor(u8).fromArray(&allocator, &inputArray0, &inputShape0);
//     defer inputTensor0.deinit();

//     // Initialize indices tensor: [0, 2]
//     var indicesArray0: [2]usize = [_]usize{ 0, 2 };
//     var indicesShape0: [1]usize = [_]usize{2};
//     var indicesTensor0 = try Tensor(usize).fromArray(&allocator, &indicesArray0, &indicesShape0);
//     defer indicesTensor0.deinit();

//     // Perform gather along axis 0
//     var gatheredTensor0 = try TensMath.gather(u8, &inputTensor0, &indicesTensor0, 0);
//     defer gatheredTensor0.deinit();

//     // Expected output tensor: [1,2,3,7,8,9], shape [2,3]
//     const expectedData0: [6]u8 = [_]u8{ 1, 2, 3, 7, 8, 9 };
//     const expectedShape0: [2]usize = [_]usize{ 2, 3 };

//     // Check shape
//     try std.testing.expect(gatheredTensor0.shape.len == expectedShape0.len);
//     for (0..expectedShape0.len) |i| {
//         try std.testing.expect(gatheredTensor0.shape[i] == expectedShape0[i]);
//     }

//     // Check data
//     try std.testing.expect(gatheredTensor0.size == 6);
//     for (0..gatheredTensor0.size) |i| {
//         try std.testing.expect(gatheredTensor0.data[i] == expectedData0[i]);
//     }

//     // -------------------------------------------------------------------------------------
//     // Test Case 2: Gather Along Axis 1
//     // -------------------------------------------------------------------------------------
//     tests_log.info("\n     test: gather along axis 1", .{});

//     var inputArray1: [2][4]u8 = [_][4]u8{
//         [_]u8{ 10, 20, 30, 40 },
//         [_]u8{ 50, 60, 70, 80 },
//     };
//     var inputShape1: [2]usize = [_]usize{ 2, 4 };
//     var inputTensor1 = try Tensor(u8).fromArray(&allocator, &inputArray1, &inputShape1);
//     defer inputTensor1.deinit();

//     var indicesArray1: [2][2]usize = [_][2]usize{
//         [_]usize{ 1, 3 },
//         [_]usize{ 0, 2 },
//     };
//     var indicesShape1: [2]usize = [_]usize{ 2, 2 };
//     var indicesTensor1 = try Tensor(usize).fromArray(&allocator, &indicesArray1, &indicesShape1);
//     defer indicesTensor1.deinit();

//     // Perform gather along axis 1
//     var gatheredTensor1 = try TensMath.gather(u8, &inputTensor1, &indicesTensor1, 1);
//     defer gatheredTensor1.deinit();

//     // Expected output tensor: [
//     //   [20, 40],
//     //   [10, 30],
//     //   [60, 80],
//     //   [50, 70]
//     // ], shape [2, 2, 2]
//     const expectedData1: [8]u8 = [_]u8{ 20, 40, 10, 30, 60, 80, 50, 70 };
//     const expectedShape1: [3]usize = [_]usize{ 2, 2, 2 };

//     // Check shape
//     try std.testing.expect(gatheredTensor1.shape.len == expectedShape1.len);
//     for (0..expectedShape1.len) |i| {
//         try std.testing.expect(gatheredTensor1.shape[i] == expectedShape1[i]);
//     }

//     // Check data
//     tests_log.debug("\n     gatheredTensor1.size: {}\n", .{gatheredTensor1.size});
//     gatheredTensor1.print();

//     try std.testing.expect(gatheredTensor1.size == 8);
//     for (0..gatheredTensor1.size) |i| {
//         tests_log.debug("\n     gatheredTensor1.data[i]: {}\n", .{expectedData1[i]});
//         tests_log.debug("\n     expectedData1[i]: {}\n", .{gatheredTensor1.data[i]});
//         try std.testing.expect(gatheredTensor1.data[i] == expectedData1[i]);
//     }

//     // -------------------------------------------------------------------------------------
//     // Test Case 3: Error Handling - Invalid Axis
//     // -------------------------------------------------------------------------------------
//     tests_log.info("\n     test: gather with invalid axis", .{});
//     const invalidAxis: usize = 3; // Input tensor has 2 dimensions
//     const result0 = TensMath.gather(u8, &inputTensor0, &indicesTensor0, invalidAxis);
//     try std.testing.expect(result0 == TensorError.InvalidAxis);
// }

// test "gather - negative axis" {
//     tests_log.info("\n     test: gather - negative axis", .{});
//     const allocator = pkgAllocator.allocator;

//     // Initialize input tensor: 2x3 matrix
//     var inputArray: [2][3]u8 = [_][3]u8{
//         [_]u8{ 1, 2, 3 },
//         [_]u8{ 4, 5, 6 },
//     };
//     var inputShape: [2]usize = [_]usize{ 2, 3 };
//     var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
//     defer inputTensor.deinit();

//     // Initialize indices tensor: [1]
//     var indicesArray: [1]usize = [_]usize{1};
//     var indicesShape: [1]usize = [_]usize{1};
//     var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
//     defer indicesTensor.deinit();

//     // Gather along axis -2 (equivalent to axis 0)
//     var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, -2);
//     defer gatheredTensor.deinit();

//     // Expected: [4, 5, 6]
//     try std.testing.expect(gatheredTensor.shape[0] == 1);
//     try std.testing.expect(gatheredTensor.shape[1] == 3);
//     try std.testing.expect(gatheredTensor.data[0] == 4);
//     try std.testing.expect(gatheredTensor.data[1] == 5);
//     try std.testing.expect(gatheredTensor.data[2] == 6);
// }

// test "gather - invalid indices" {
//     tests_log.info("\n     test: gather - invalid indices", .{});
//     const allocator = pkgAllocator.allocator;

//     // Initialize input tensor: 2x2 matrix
//     var inputArray: [2][2]u8 = [_][2]u8{
//         [_]u8{ 1, 2 },
//         [_]u8{ 3, 4 },
//     };
//     var inputShape: [2]usize = [_]usize{ 2, 2 };
//     var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
//     defer inputTensor.deinit();

//     // Initialize indices tensor with invalid index
//     var indicesArray: [1]usize = [_]usize{2}; // Invalid index (only 0,1 are valid)
//     var indicesShape: [1]usize = [_]usize{1};
//     var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
//     defer indicesTensor.deinit();

//     // Should return error for out of bounds index
//     try std.testing.expectError(TensorError.IndexOutOfBounds, TensMath.gather(u8, &inputTensor, &indicesTensor, 0));
// }

// test "gather - multi-dimensional indices" {
//     tests_log.info("\n     test: gather - multi-dimensional indices", .{});
//     const allocator = pkgAllocator.allocator;

//     // Initialize input tensor: 3x3 matrix
//     var inputArray: [3][3]u8 = [_][3]u8{
//         [_]u8{ 1, 2, 3 },
//         [_]u8{ 4, 5, 6 },
//         [_]u8{ 7, 8, 9 },
//     };
//     var inputShape: [2]usize = [_]usize{ 3, 3 };
//     var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
//     defer inputTensor.deinit();

//     // Initialize 2D indices tensor: [[0,2], [1,1]]
//     var indicesArray: [2][2]usize = [_][2]usize{
//         [_]usize{ 0, 2 },
//         [_]usize{ 1, 1 },
//     };
//     var indicesShape: [2]usize = [_]usize{ 2, 2 };
//     var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
//     defer indicesTensor.deinit();

//     // Gather along axis 0
//     var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, 0);
//     defer gatheredTensor.deinit();

//     // Expected shape: [2, 2, 3]
//     try std.testing.expect(gatheredTensor.shape[0] == 2);
//     try std.testing.expect(gatheredTensor.shape[1] == 2);
//     try std.testing.expect(gatheredTensor.shape[2] == 3);

//     // Check first row (indices 0,2): [1,2,3], [7,8,9]
//     try std.testing.expect(gatheredTensor.data[0] == 1);
//     try std.testing.expect(gatheredTensor.data[1] == 2);
//     try std.testing.expect(gatheredTensor.data[2] == 3);
//     try std.testing.expect(gatheredTensor.data[3] == 7);
//     try std.testing.expect(gatheredTensor.data[4] == 8);
//     try std.testing.expect(gatheredTensor.data[5] == 9);

//     // Check second row (indices 1,1): [4,5,6], [4,5,6]
//     try std.testing.expect(gatheredTensor.data[6] == 4);
//     try std.testing.expect(gatheredTensor.data[7] == 5);
//     try std.testing.expect(gatheredTensor.data[8] == 6);
//     try std.testing.expect(gatheredTensor.data[9] == 4);
//     try std.testing.expect(gatheredTensor.data[10] == 5);
//     try std.testing.expect(gatheredTensor.data[11] == 6);
// }

// test "gather - single element tensor" {
//     tests_log.info("\n     test: gather - single element tensor", .{});
//     const allocator = pkgAllocator.allocator;

//     // Initialize input tensor: [[[1]]]
//     var inputArray: [1][1][1]u8 = [_][1][1]u8{[_][1]u8{[_]u8{1}}};
//     var inputShape: [3]usize = [_]usize{ 1, 1, 1 };
//     var inputTensor = try Tensor(u8).fromArray(&allocator, &inputArray, &inputShape);
//     defer inputTensor.deinit();

//     // Initialize indices tensor: [0]
//     var indicesArray: [1]usize = [_]usize{0};
//     var indicesShape: [1]usize = [_]usize{1};
//     var indicesTensor = try Tensor(usize).fromArray(&allocator, &indicesArray, &indicesShape);
//     defer indicesTensor.deinit();

//     // Test gathering on each axis
//     inline for (0..3) |axis| {
//         var gatheredTensor = try TensMath.gather(u8, &inputTensor, &indicesTensor, axis);
//         defer gatheredTensor.deinit();

//         try std.testing.expect(gatheredTensor.data[0] == 1);
//         try std.testing.expect(gatheredTensor.size == 1);
//     }
// }
