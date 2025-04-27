const testing = std.testing;

const std = @import("std");

const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const TensorProto = zant.onnx.TensorProto;
const Tensor = zant.core.Tensor;

const ProtoTensor2Tensor = @import("ProtoTensor2Tensor.zig").ProtoTensor2Tensor;

// Test for raw data not available
test "ProtoTensor2Tensor: float32 parsing" {
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .FLOAT,
        .float_data = &values,
        .raw_data = null,
    };

    const tensor = try ProtoTensor2Tensor(f32, proto);
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
    const values = [_]i32{ 10, 20, 30, 40 };
    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .INT32,
        .int32_data = &values,
        .raw_data = null,
    };

    const tensor = try ProtoTensor2Tensor(i32, proto, allocator);
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
    const values = [_]i64{ 100, 200, 300, 400 };

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .INT64,
        .int64_data = &values,
        .raw_data = null,
    };

    const tensor = try ProtoTensor2Tensor(i64, proto, allocator);
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
    const values = [_]f64{ 1.23456789, 2.34567891, 3.45678901, 4.56789012 };

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .FLOAT64,
        .int64_data = &values,
        .raw_data = null,
    };

    const tensor = try ProtoTensor2Tensor(i64, proto, allocator);
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
    const values = [_]u64{ 1234567890123456, 9876543219876543, 1234567899876543, 9999999999999999 };

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .UINT64,
        .int64_data = &values,
        .raw_data = null,
    };

    const tensor = try ProtoTensor2Tensor(i64, proto, allocator);
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

// Test for raw data available
test "ProtoTensor2Tensor: raw data parsing float32" {
    const raw_values = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    const raw_bytes = std.mem.sliceAsBytes(&raw_values);
    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .FLOAT,
        .float_data = null,
        .raw_data = raw_bytes,
    };

    const tensor = try ProtoTensor2Tensor(f32, proto, allocator);
    defer tensor.deinit();
    // test size
    try testing.expectEqual(@as(usize, 4), tensor.size);
    // test data
    try testing.expectEqual(1.5, tensor.data[0]);
    try testing.expectEqual(2.5, tensor.data[1]);
    try testing.expectEqual(3.5, tensor.data[2]);
    try testing.expectEqual(4.5, tensor.data[3]);
    // test shape
    try testing.expectEqual(2, tensor.shape[0]);
    try testing.expectEqual(2, tensor.shape[1]);
}

test "ProtoTensor2Tensor: raw data parsing int32" {
    const raw_values = [_]i32{ 1000, 2000, 3000, 4000 };
    const raw_bytes = std.mem.sliceAsBytes(&raw_values);

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .INT32,
        .int32_data = null,
        .raw_data = raw_bytes,
    };

    const tensor = try ProtoTensor2Tensor(i32, proto, allocator);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(i32, 1000), tensor.data[0]);
    try testing.expectEqual(@as(i32, 4000), tensor.data[3]);
}

test "ProtoTensor2Tensor: raw data parsing int64" {
    const raw_values = [_]i64{ 10000000000, 20000000000, 30000000000, 40000000000 };
    const raw_bytes = std.mem.sliceAsBytes(&raw_values);

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .INT64,
        .int32_data = null,
        .raw_data = raw_bytes,
    };

    const tensor = try ProtoTensor2Tensor(i64, proto, allocator);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(i64, 10000000000), tensor.data[0]);
    try testing.expectEqual(@as(i64, 40000000000), tensor.data[3]);
}

test "ProtoTensor2Tensor: raw data parsing float64" {
    const raw_values = [_]f64{ 1.5, 2.5, 3.5, 4.5 };
    const raw_bytes = std.mem.sliceAsBytes(&raw_values);

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .FLOAT64,
        .float_data = null,
        .raw_data = raw_bytes,
    };

    const tensor = try ProtoTensor2Tensor(f64, proto, allocator);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(1.5, tensor.data[0]);
    try testing.expectEqual(2.5, tensor.data[1]);
    try testing.expectEqual(3.5, tensor.data[2]);
    try testing.expectEqual(4.5, tensor.data[3]);
}

test "ProtoTensor2Tensor: raw data parsing uint64" {
    const raw_values = [_]u64{ 10000000000, 20000000000, 30000000000, 40000000000 };
    const raw_bytes = std.mem.sliceAsBytes(&raw_values);

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .UINT64,
        .raw_data = raw_bytes,
        .int32_data = null,
    };

    const tensor = try ProtoTensor2Tensor(u64, proto, allocator);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(u64, 10000000000), tensor.data[0]);
    try testing.expectEqual(@as(u64, 40000000000), tensor.data[3]);
}

test "ProtoTensor2Tensor: raw data parsing uint16" {
    const raw_values = [_]u16{ 1000, 2000, 3000, 4000 };
    const raw_bytes = std.mem.sliceAsBytes(&raw_values);

    const proto = TensorProto{
        .dims = &.{ 2, 2 },
        .data_type = .UINT16,
        .raw_data = raw_bytes,
    };

    const tensor = try ProtoTensor2Tensor(u16, proto, allocator);
    defer tensor.deinit();

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(u16, 1000), tensor.data[0]);
    try testing.expectEqual(@as(u16, 4000), tensor.data[3]);
}
