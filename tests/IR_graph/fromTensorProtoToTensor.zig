const testing = std.testing;

const std = @import("std");

const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const TensorProto = zant.onnx.TensorProto;
const Tensor = zant.core.Tensor;

const ProtoTensor2Tensor = @import("ProtoTensor2Tensor.zig").ProtoTensor2Tensor;

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

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectApproxEqAbs(f32, 1.0, tensor.data[0], 0.001);
    try testing.expectApproxEqAbs(f32, 4.0, tensor.data[3], 0.001);
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

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(i32, 10), tensor.data[0]);
    try testing.expectEqual(@as(i32, 40), tensor.data[3]);
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

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectEqual(@as(i64, 100), tensor.data[0]);
    try testing.expectEqual(@as(i64, 400), tensor.data[3]);
}

test "ProtoTensor2Tensor: raw data parsing float" {
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

    try testing.expectEqual(@as(usize, 4), tensor.size);
    try testing.expectApproxEqAbs(f32, 1.5, tensor.data[0], 0.001);
    try testing.expectApproxEqAbs(f32, 4.5, tensor.data[3], 0.001);
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
