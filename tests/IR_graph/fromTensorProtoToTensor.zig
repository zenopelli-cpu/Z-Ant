const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const TensorProto = zant.onnx.TensorProto;
const Tensor = zant.core.tensor.Tensor;

const ProtoTensor2Tensor = zant.IR_graph.NodeZant.ProtoTensor2Tensor;

// Test for raw data not available
test "ProtoTensor2Tensor: float32 parsing" {
    var dims = [_]i64{ 2, 2 };
    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    const proto = TensorProto{
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

    var tensor = try ProtoTensor2Tensor(f32, proto);
    defer tensor.deinit();
    // test size
    try testing.expectEqual(@as(usize, 4), tensor.size);
    // test data
    try testing.expectEqual(1.0, tensor.data[0]);
    try testing.expectEqual(2.0, tensor.data[1]);
    try testing.expectEqual(3.0, tensor.data[2]);
    try testing.expectEqual(4.0, tensor.data[3]);
    // test shape
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

test "ProtoTensor2Tensor: int32 parsing" {
    var dims = [_]i64{ 2, 2 };
    var values = [_]i32{ 10, 20, 30, 40 };

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .INT32,
        .int32_data = &values,
        .raw_data = null,

        //data not useful for test
        .segment = null,
        .name = null,
        .float_data = null,
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

    var tensor = try ProtoTensor2Tensor(i32, proto);
    defer tensor.deinit();

    // test size
    try testing.expectEqual(@as(usize, 4), tensor.size);
    // test data
    try testing.expectEqual(@as(i32, 10), tensor.data[0]);
    try testing.expectEqual(@as(i32, 20), tensor.data[1]);
    try testing.expectEqual(@as(i32, 30), tensor.data[2]);
    try testing.expectEqual(@as(i32, 40), tensor.data[3]);
    // test shape
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

test "ProtoTensor2Tensor: int64 parsing" {
    var dims = [_]i64{ 2, 2 };
    var values = [_]i64{ 100, 200, 300, 400 };

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .INT64,
        .int64_data = &values,
        .raw_data = null,

        //data not useful for test
        .segment = null,
        .name = null,
        .int32_data = null,
        .float_data = null,
        .string_data = null,
        .double_data = null,
        .uint64_data = null,
        .uint16_data = null,
        .doc_string = null,
        .external_data = undefined,
        .data_location = null,
        .metadata_props = undefined,
    };

    var tensor = try ProtoTensor2Tensor(i64, proto);
    defer tensor.deinit();

    // test size
    try testing.expectEqual(@as(usize, 4), tensor.size);
    // test data
    try testing.expectEqual(@as(i64, 100), tensor.data[0]);
    try testing.expectEqual(@as(i64, 200), tensor.data[1]);
    try testing.expectEqual(@as(i64, 300), tensor.data[2]);
    try testing.expectEqual(@as(i64, 400), tensor.data[3]);
    // test shape
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

test "ProtoTensor2Tensor: float64 parsing" {
    var dims = [_]i64{ 2, 2 };
    var values = [_]f64{ 1.23456789, 2.34567891, 3.45678901, 4.56789012 };

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .FLOAT,
        .double_data = &values,
        .raw_data = null,

        //data not useful for test
        .segment = null,
        .name = null,
        .int32_data = null,
        .int64_data = null,
        .float_data = null,
        .string_data = null,
        .uint64_data = null,
        .uint16_data = null,
        .doc_string = null,
        .external_data = undefined,
        .data_location = null,
        .metadata_props = undefined,
    };

    var tensor = try ProtoTensor2Tensor(f64, proto);
    defer tensor.deinit();

    // test size
    try testing.expectEqual(@as(usize, 4), tensor.size);
    // test data
    try testing.expectEqual(1.23456789, tensor.data[0]);
    try testing.expectEqual(2.34567891, tensor.data[1]);
    try testing.expectEqual(3.45678901, tensor.data[2]);
    try testing.expectEqual(4.56789012, tensor.data[3]);
    // test shape
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

test "ProtoTensor2Tensor: unit64 parsing" {
    var dims = [_]i64{ 2, 2 };
    var values = [_]u64{ 1234567890123456, 9876543219876543, 1234567899876543, 9999999999999999 };

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .UINT64,
        .uint64_data = &values,
        .raw_data = null,

        //data not useful for test
        .segment = null,
        .name = null,
        .int32_data = null,
        .int64_data = null,
        .float_data = null,
        .string_data = null,
        .double_data = null,
        .uint16_data = null,
        .doc_string = null,
        .external_data = undefined,
        .data_location = null,
        .metadata_props = undefined,
    };

    var tensor = try ProtoTensor2Tensor(u64, proto);
    defer tensor.deinit();

    // test size
    try testing.expectEqual(@as(usize, 4), tensor.size);
    // test data
    try testing.expectEqual(1234567890123456, tensor.data[0]);
    try testing.expectEqual(9876543219876543, tensor.data[1]);
    try testing.expectEqual(1234567899876543, tensor.data[2]);
    try testing.expectEqual(9999999999999999, tensor.data[3]);
    // test shape
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}
