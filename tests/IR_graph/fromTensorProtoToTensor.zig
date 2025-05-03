const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const TensorProto = zant.onnx.TensorProto;
const Tensor = zant.core.tensor.Tensor;

const protoTensor2Tensor = zant.IR_graph.NodeZant.protoTensor2Tensor;
const getType = zant.IR_graph.NodeZant.getType;

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

    var tensor = try protoTensor2Tensor(f32, proto);
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

    var tensor = try protoTensor2Tensor(i32, proto);
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

    var tensor = try protoTensor2Tensor(i64, proto);
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

    var tensor = try protoTensor2Tensor(f64, proto);
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

    var tensor = try protoTensor2Tensor(u64, proto);
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
// Float32
test "ProtoTensor2Tensor: float32 parsing from raw_data" {
    var dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const raw_bytes = std.mem.sliceAsBytes(&values);

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .FLOAT,
        .raw_data = raw_bytes,
        .float_data = null,

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

    var tensor = try protoTensor2Tensor(f32, proto);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(1.0, tensor.data[0]);
    try testing.expectEqual(2.0, tensor.data[1]);
    try testing.expectEqual(3.0, tensor.data[2]);
    try testing.expectEqual(4.0, tensor.data[3]);
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

// Int32
test "ProtoTensor2Tensor: int32 parsing from raw_data" {
    var dims = [_]i64{ 2, 2 };
    const values = [_]i32{ 10, 20, 30, 40 };
    const raw_bytes = std.mem.sliceAsBytes(&values);

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .INT32,
        .raw_data = raw_bytes,
        .int32_data = null,

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

    var tensor = try protoTensor2Tensor(i32, proto);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(i32, 10), tensor.data[0]);
    try testing.expectEqual(@as(i32, 20), tensor.data[1]);
    try testing.expectEqual(@as(i32, 30), tensor.data[2]);
    try testing.expectEqual(@as(i32, 40), tensor.data[3]);
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

// Int64
test "ProtoTensor2Tensor: int64 parsing from raw_data" {
    var dims = [_]i64{ 2, 2 };
    const values = [_]i64{ 100, 200, 300, 400 };
    const raw_bytes = std.mem.sliceAsBytes(&values);

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .INT64,
        .raw_data = raw_bytes,
        .int64_data = null,

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

    var tensor = try protoTensor2Tensor(i64, proto);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(i64, 100), tensor.data[0]);
    try testing.expectEqual(@as(i64, 200), tensor.data[1]);
    try testing.expectEqual(@as(i64, 300), tensor.data[2]);
    try testing.expectEqual(@as(i64, 400), tensor.data[3]);
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

// Float64
test "ProtoTensor2Tensor: float64 parsing from raw_data" {
    var dims = [_]i64{ 2, 2 };
    const values = [_]f64{ 1.23456789, 2.34567891, 3.45678901, 4.56789012 };
    const raw_bytes = std.mem.sliceAsBytes(&values);

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .DOUBLE,
        .raw_data = raw_bytes,
        .double_data = null,

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

    var tensor = try protoTensor2Tensor(f64, proto);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(1.23456789, tensor.data[0]);
    try testing.expectEqual(2.34567891, tensor.data[1]);
    try testing.expectEqual(3.45678901, tensor.data[2]);
    try testing.expectEqual(4.56789012, tensor.data[3]);
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

// Uint64
test "ProtoTensor2Tensor: uint64 parsing from raw_data" {
    var dims = [_]i64{ 2, 2 };
    const values = [_]u64{
        1234567890123456,
        9876543219876543,
        1234567899876543,
        9999999999999999,
    };
    const raw_bytes = std.mem.sliceAsBytes(&values);

    const proto = TensorProto{
        .dims = &dims,
        .data_type = .UINT64,
        .raw_data = raw_bytes,
        .uint64_data = null,

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

    var tensor = try protoTensor2Tensor(u64, proto);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(1234567890123456, tensor.data[0]);
    try testing.expectEqual(9876543219876543, tensor.data[1]);
    try testing.expectEqual(1234567899876543, tensor.data[2]);
    try testing.expectEqual(9999999999999999, tensor.data[3]);
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}
