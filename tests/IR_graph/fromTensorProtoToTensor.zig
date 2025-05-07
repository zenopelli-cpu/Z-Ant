const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;

const TensorProto = zant.onnx.TensorProto;
const Tensor = zant.core.tensor.Tensor;

const protoTensor2Tensor = zant.IR_graph.TensorZant.protoTensor2Tensor;

// Test for raw data not available
test "ProtoTensor2Tensor: float32 parsing" {
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

    var anyTensor = try protoTensor2Tensor(&proto);
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
