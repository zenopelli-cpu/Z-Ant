const std = @import("std");
const zant = @import("../../../zant.zig");
const quant = zant.core.quantization;
const Tensor = zant.core.tensor.Tensor;
const TensorType = zant.core.tensor.TensorType;

const pkgAllocator = zant.utils.allocator.allocator;

pub const quantScheme = enum {
    SYMM,
    ASYM,
};

/// Quantizes the input Tensor to the outputType.
/// Returns the quantized Tensor.
pub fn dequantize(comptime inputType: type, comptime outputType: type, input: *Tensor(inputType)) !Tensor(outputType) {
    const output_shape = try get_dequantize_output_shape(input.shape);
    defer pkgAllocator.free(output_shape);

    var output = try Tensor(outputType).fromShape(&pkgAllocator, output_shape);
    errdefer output.deinit();

    try lean_dequantize(inputType, outputType, input, &output);

    return output;
}

pub fn lean_dequantize(comptime inputType: type, comptime outputType: type, input: *Tensor(inputType), output: *Tensor(outputType)) !void {
    const scale = try input.get_scale_factor();
    const zero = try input.get_zero_point();
    const dequantizedArray = try dequantize_array(inputType, outputType, input.data, scale, zero);
    defer pkgAllocator.free(dequantizedArray);

    @memcpy(output.data, dequantizedArray);

    output.details = .{ .none = {} };
}

pub fn get_dequantize_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkgAllocator.alloc(usize, input_shape.len);
    errdefer pkgAllocator.free(output_shape);

    @memcpy(output_shape, input_shape);

    return output_shape;
}

// ========== dequantization

/// This function dequantizes the monodimensional input array, using the given parameters:
/// the current unquantized type T, the quantized output type U, the input array, the scale factor, the zero point.
/// The caller is responsible for freeing the returned array.
pub fn dequantize_array(comptime inputType: type, comptime outputType: type, inputArray: []const inputType, scale: f32, zero: isize) ![]outputType {
    var output = try pkgAllocator.alloc(outputType, inputArray.len);

    for (inputArray, 0..) |val, i| {
        // dequantize each val
        const correctedVal: isize = @as(isize, val) - @as(isize, zero);
        output[i] = scale * @as(outputType, @floatFromInt(correctedVal));
    }

    return output;
}
