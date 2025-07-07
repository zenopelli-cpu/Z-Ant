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
/// Parameters:
/// - inputType: data type of the tensor to be quantized.
/// - outputType: data type to be quantized to.
/// - input: pointer to the input tensor to be quantized
/// - scheme: quantization scheme (symmetric or asymmetric)
/// note: as of now the quantization scheme is ignored and hardcoded to asymmetric
/// Returns the quantized Tensor.
pub fn quantize(comptime inputType: type, comptime outputType: type, input: *Tensor(inputType), scheme: quantScheme) !Tensor(outputType) {
    const hardcodedScheme = quantScheme.ASYM; // asymm hardcoded
    _ = scheme;

    const output_shape = try get_quantize_output_shape(input.shape);
    defer pkgAllocator.free(output_shape);

    var output = try Tensor(outputType).fromShape(&pkgAllocator, output_shape);
    errdefer output.deinit();

    // minmax quantization "hardcoded"
    try lean_quantize_minmax(inputType, outputType, input, &output, hardcodedScheme);

    return output;
}

pub fn lean_quantize_minmax(comptime inputType: type, comptime outputType: type, input: *Tensor(inputType), output: *Tensor(outputType), scheme: quantScheme) !void {
    const result = try minmax_array_quant(inputType, outputType, scheme, input.data);
    defer pkgAllocator.free(result.quantizedArray);

    @memcpy(output.data, result.quantizedArray);

    output.details = .{ .quant = .{
        .tensorType = TensorType.QuantTensor,
        .scale_factor = result.scale,
        .zero_point = result.zero,
    } };
}

pub fn get_quantize_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkgAllocator.alloc(usize, input_shape.len);
    errdefer pkgAllocator.free(output_shape);

    @memcpy(output_shape, input_shape);

    return output_shape;
}

// ------------------------- Quantization methods and helper functions -------------------------

// ========== helper functions
pub fn clamp(comptime T: type, comptime U: type, value: T, scale: T, zero: i32, minInt: U, maxInt: U) U {
    var roundedVal: T = undefined;

    if(scale == 0) {
        roundedVal = value;
    } else {
        roundedVal = @round(value / scale + @as(T, @floatFromInt(zero)));
    }

    if (roundedVal <= @as(T, @floatFromInt(minInt)))
        return minInt;
    if (roundedVal >= @as(T, @floatFromInt(maxInt)))
        return maxInt;

    const roundedValInt: U = @as(U, @intFromFloat(roundedVal));

    return roundedValInt;
}

pub inline fn get_scale_factor(comptime T: type, comptime U: type, minFloat: T, maxFloat: T) T {
    const num: T = maxFloat - minFloat;

    const num_elements = (1 << @bitSizeOf(U)) - 1; // 2^b - 1 values
    const denom: T = @as(T, @floatFromInt(num_elements));

    return num / denom;
}

/// Computes and returns the zero point.
/// Params:
/// - T: input type
/// - U: outputtype
/// - scale: scale factor
/// - minFloat: minimum value in the floating point range
/// - minInt: minimum value of the quantized data type range
/// Returns: the zero point as a i32
pub inline fn get_zero_point(comptime T: type, comptime U: type, scale: T, minFloat: T, minInt: U) i32 {
    if(scale == 0)
        return @as(i32, @intCast(minInt)) - @as(i32, @intFromFloat(minFloat));

    const zeroPointFloat: T = minFloat / scale;

    return @as(i32, @intCast(minInt)) - @as(i32, @intFromFloat(zeroPointFloat));
}

// ========== quantization

/// This function quantizes the input monodimensional array, using the given parameters:
/// scale factor, zero point, minInt/maxInt (aka the integer grid limits).
/// Parameters:
/// - T: input data type
/// - U: output data type, to be quantized to
/// - inputArray: array to be quantized
/// - scale: the scale factor
/// - zero: the zero point
/// - minInt: chosen min value of the quantized range
/// - maxInt: chosen max value of the quantized range
/// Returns the quantized array.
/// The caller is responsible for freeing the returned array.
pub fn quantize_array(comptime T: type, comptime U: type, inputArray: anytype, scale: T, zero: i32, minInt: U, maxInt: U) ![]U {
    var output = try pkgAllocator.alloc(U, inputArray.len);

    for (inputArray, 0..) |val, i| {
        // quantize each val
        output[i] = clamp(T, U, val, scale, zero, minInt, maxInt);
    }

    return output;
}

/// This function quantizes the input monodimensional array using min/max method.
/// Parameters:
/// - T: input data type
/// - U: output data type, to be quantized to
/// - input: array to be quantized
/// - scheme: quantization scheme (symmetric or asymmetric)
/// Returns a tuple with the result quantized array, scale factor, zero point.
/// note: as of now the quantization scheme is ignored and hardcoded to asymmetric
/// The caller is responsible for freeing the returned quantized array.
pub fn minmax_array_quant(comptime T: type, comptime U: type, scheme: quantScheme, input: anytype) !struct { quantizedArray: []U, scale: T, zero: i32 } {
    var minFloat: T = input[0];
    var maxFloat: T = input[0];

    // compute the min and max value if the input tensor
    for (input[1..]) |val| {
        if (minFloat > val)
            minFloat = val;
        if (maxFloat < val)
            maxFloat = val;
    }

    // compute minInt and maxInt
    const minInt = if (@typeInfo(U) == .int) std.math.minInt(U) else std.math.floatMin(U);
    const maxInt = if (@typeInfo(U) == .int) std.math.maxInt(U) else std.math.floatMax(U);

    const scale: T = get_scale_factor(T, U, minFloat, maxFloat);

    var zero: i32 = undefined;
    switch (scheme) {
        quantScheme.SYMM => zero = 0,
        quantScheme.ASYM => zero = get_zero_point(T, U, scale, minFloat, minInt),
    }

    const quantizedArray: []U = try quantize_array(T, U, input, scale, zero, minInt, maxInt);
    const immutableZero: i32 = zero;
    return .{
        .quantizedArray = quantizedArray,
        .scale = scale,
        .zero = immutableZero,
    };
}
