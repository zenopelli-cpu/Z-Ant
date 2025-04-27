const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;

const allocator = zant.utils.allocator.allocator;

const Tensor = zant.core.Tensor;
const TensorProto = onnx.TensorProto;
const DataType = onnx.DataType;

const TensorError = zant.utils.errorHandler.TensorError;
const TensorMathError = zant.utils.errorHandler.TensorMathError;

pub fn ProtoTensor2Tensor(comptime T: type, proto: TensorProto) !Tensor(T) {
    // Type Check
    if (!isMatchingType(T, proto.data_type)) {
        return TensorError.InputArrayWrongType;
    }

    // Allocate shape array
    var shape = try allocator.alloc(usize, proto.dims.len);
    for (proto.dims, 0..) |dim, i| {
        if (dim < 0) {
            return TensorMathError.InvalidDimensions;
        }
        shape[i] = @intCast(dim);
    }

    // Compute total size
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }

    // Allocate data array
    const data = try allocator.alloc(T, size);

    // Fill data
    if (proto.raw_data) |raw| {
        // Fill from raw_data
        const needed_bytes = size * @sizeOf(T);
        if (raw.len != needed_bytes) {
            return error.RawDataSizeMismatch;
        }
        std.mem.copy(u8, @ptrCast(data.ptr), raw);
    } else {
        // Fill from typed fields
        if (T == f32 and proto.float_data) |floats| {
            if (floats.len != size) return TensorMathError.InputTensorDifferentSize;
            std.mem.copy(T, data.ptr, floats);
        } else if (T == i32 and proto.int32_data) |ints| {
            if (ints.len != size) return TensorMathError.InputTensorDifferentSize;
            std.mem.copy(T, data.ptr, ints);
        } else if (T == i64 and proto.int64_data) |ints| {
            if (ints.len != size) return TensorMathError.InputTensorDifferentSize;
            std.mem.copy(T, data.ptr, ints);
        } else if (T == f64 and proto.double_data) |doubles| {
            if (doubles.len != size) return TensorMathError.InputTensorDifferentSize;
            std.mem.copy(T, data.ptr, doubles);
        } else if (T == u64 and proto.uint64_data) |uints| {
            if (uints.len != size) return TensorMathError.InputTensorDifferentSize;
            std.mem.copy(T, data.ptr, uints);
        } else if (T == u16 and proto.uint16_data) |uints| {
            if (uints.len != size) return TensorMathError.InputTensorDifferentSize;
            std.mem.copy(T, data.ptr, uints);
        } else {
            return TensorError.NanValue;
        }
    }

    // Return the Tensor
    return Tensor(T){
        .data = data,
        .size = size,
        .shape = shape,
        .allocator = allocator,
    };
}

fn isMatchingType(comptime T: type, data_type: DataType) bool {
    return switch (data_type) {
        .FLOAT => T == f32,
        .INT32 => T == i32,
        .INT64 => T == i64,
        .DOUBLE => T == f64,
        .UINT64 => T == u64,
        .UINT16 => T == u16,
        else => false,
    };
}
