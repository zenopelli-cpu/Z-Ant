const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("../../../zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;

// https://onnx.ai/onnx/operators/onnx__Gemm.html
// INPUTS:
//      - Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
//      - Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
//      - Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
//OUTPUTS:
//      - Output tensor of shape (M, N).
// ATTRIBUTES:
//      - alpha. FLOAT (default is '1.0'): Scalar multiplier for the product of input tensors A * B.
//      - beta - FLOAT (default is '1.0'): Scalar multiplier for input tensor C.
//      - transA - INT (default is '0'): Whether A should be transposed
//      - transB - INT (default is '0'): Whether B should be transposed

pub const Gemm = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    input_C: ?*TensorZant,
    output: *TensorZant,
    //attributes:
    alpha: f32, // = 1.0;
    beta: f32, // = 1.0;
    transA: bool, // = false;
    transB: bool, // = false;

    pub fn init(nodeProto: *NodeProto) !Gemm {
        const input_A = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const input_C = if (nodeProto.input.len > 2) if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_C_notFound else null;
        const output = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 1.0;
        var beta: f32 = 1.0;
        var transA: bool = false;
        var transB: bool = false;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "alpha")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) alpha = attr.f else return error.GemmAphaNotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "beta")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) beta = attr.f else return error.GemmBetaNotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "transA")) |_| {
                if (attr.type == onnx.AttributeType.INT) transA = if (attr.i != 0) true else false else return error.GemmTransANotINT;
            } else if (std.mem.indexOf(u8, attr.name, "transB")) |_| {
                if (attr.type == onnx.AttributeType.INT) transB = if (attr.i != 0) true else false else return error.GemmTransBNotINT;
            }
        }

        return Gemm{
            .input_A = input_A,
            .input_B = input_B,
            .input_C = input_C,
            .output = output,
            .alpha = alpha,
            .beta = beta,
            .transA = transA,
            .transB = transB,
        };
    }

    pub fn get_output_shape(self: Gemm) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn print(self: Gemm) void { // TODO
        std.debug.print("\n Gemm:\n {any}", .{self});
    }
};
