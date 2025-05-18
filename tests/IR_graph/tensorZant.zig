const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const TensorZant = zant.IR_graph.TensorZant;
const allocator = zant.utils.allocator.allocator;

const TensorProto = zant.onnx.TensorProto;
const Tensor = zant.core.tensor.Tensor;

const protoTensor2AnyTensor = zant.IR_graph.utils.protoTensor2AnyTensor;

// Test for raw data not available
test "protoTensor2AnyTensor: float32 parsing" {
    std.debug.print("\n\n ------TEST: protoTensor2AnyTensor: float32 parsing", .{});

    var dims = [_]i64{ 2, 2 };
    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var proto = TensorProto{
        .dims = &dims,
        .data_type = .FLOAT,
        .float_data = &values,
        .raw_data = null,

        //data not useful for test
        .segment = null,
        .name = null,
        .int32_data = null,
        .string_data = null,
        .int64_data = null,
        .double_data = null,
        .uint64_data = null,
        .uint16_data = null,
        .doc_string = null,
        .external_data = undefined,
        .data_location = null,
        .metadata_props = undefined,
    };

    var anyTensor = try protoTensor2AnyTensor(&proto);
    defer anyTensor.deinit();

    // test size
    try testing.expectEqual(@as(usize, 4), anyTensor.f32.size);
    // test data
    try testing.expectEqual(1.0, anyTensor.f32.data[0]);
    try testing.expectEqual(2.0, anyTensor.f32.data[1]);
    try testing.expectEqual(3.0, anyTensor.f32.data[2]);
    try testing.expectEqual(4.0, anyTensor.f32.data[3]);
    // test shape
    try testing.expectEqual(2, anyTensor.f32.shape[0]);
    try testing.expectEqual(2, anyTensor.f32.shape[1]);
}

test "computeStride with 3D shape" {
    std.debug.print("\n\n ------TEST: computeStride with 3D shape", .{});

    var shape = [_]usize{ 2, 3, 4 };
    const expected = [_]usize{ 12, 4, 1 };

    const strides = try TensorZant.computeStride(&shape);

    try std.testing.expectEqualSlices(usize, &expected, strides);
}

test "computeStride with 2D shape" {
    std.debug.print("\n\n ------TEST: computeStride with 2D shape", .{});

    var shape = [_]usize{ 5, 10 };
    const expected = [_]usize{ 10, 1 };

    const strides = try TensorZant.computeStride(&shape);

    try std.testing.expectEqualSlices(usize, &expected, strides);
}

test "computeStride with 1D shape" {
    std.debug.print("\n\n ------TEST: computeStride with 1D shape", .{});

    var shape = [_]usize{7};
    const expected = [_]usize{1};

    const strides = try TensorZant.computeStride(&shape);

    try std.testing.expectEqualSlices(usize, &expected, strides);
}
