const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

//----------------- LOG OPERATOR ------------------------

pub fn get_log_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

//output con un tensore  (no lean version)
pub fn log(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    if (isLogSupportedType(T)) {
        return TensorMathError.InvalidDataType;
    }

    if (input.data.len == 0) {
        return TensorError.ZeroSizeTensor;
    }

    const output_shape = try get_log_output_shape(input.shape);
    defer pkg_allocator.free(output_shape);

    var outputTensor = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer outputTensor.deinit();

    try log_lean(T, input, &outputTensor);

    return outputTensor;
}

//output void (lean version)
pub inline fn log_lean(comptime T: type, input: *const Tensor(T), output: *Tensor(T)) !void {
    const input_data = input.data;
    const output_data = output.data;

    if (input_data.len != output_data.len) {
        return TensorError.OutputTensorWrongShape;
    }

    if (T == zant.core.types.bfloat16) {
        for (input_data, output_data) |x, *y| {
            const xf32: f32 = bf16_to_f32(x);
            if (xf32 <= 0.0) {
                return TensorMathError.InvalidDataType;
            }
            const yf32: f32 = @log(xf32);
            y.* = f32_to_bf16(yf32);
        }
    } else {
        for (input_data, output_data) |x, *y| {
            if (x <= 0) {
                return TensorMathError.InvalidDataType;
            }
            y.* = @log(x);
        }
    }
}

//check se il tipo di tensore è accettato o meno
fn isLogSupportedType(comptime T: type) bool {
    return switch (T) {
        f16, f32, f64 => false,
        zant.core.types.bfloat16 => false,
        else => true,
    };
}

//----------------- BF16 <-> f32 HELPERS -----------------

const BF16 = zant.core.types.bfloat16;

// Estrai/inserisci i bit (adatta se il tuo BF16 ha nome/campo diverso)
// Se BF16 è un alias di u16 o uno struct { bits: u16 }:
inline fn bf16_get_bits(x: BF16) u16 {
    return switch (@typeInfo(BF16)) {
        .Int => @intCast(x), // 1 arg, il target è u16 (tipo della funzione)
        .Struct => x.bits,
        else => @compileError("bfloat16 non supportato"),
    };
}

inline fn bf16_from_bits(bits: u16) BF16 {
    return switch (@typeInfo(BF16)) {
        .Int => @as(BF16, bits), // conversione al tipo BF16
        .Struct => BF16{ .bits = bits },
        else => @compileError("bfloat16 non supportato"),
    };
}

// bfloat16 -> f32
fn bf16_to_f32(x: BF16) f32 {
    const hi: u32 = @as(u32, bf16_get_bits(x)) << 16;
    return @bitCast(hi); // 1 arg: ritorna f32 perché la funzione ritorna f32
}

// f32 -> bfloat16 (round-to-nearest-even)
fn f32_to_bf16(x: f32) BF16 {
    const u: u32 = @bitCast(x); // 1 arg
    const lsb: u32 = (u >> 16) & 1;
    const rounded: u32 = u + 0x7FFF + lsb;
    const hi: u16 = @intCast(rounded >> 16); // 1 arg, target è u16
    return bf16_from_bits(hi);
}
