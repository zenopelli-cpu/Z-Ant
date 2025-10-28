const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

// --- uops ---
const cg_v2 = @import("codegen").codegen_v2;
const Uops = cg_v2.uops;
const UOpBuilder = cg_v2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

// https://onnx.ai/onnx/operators/onnx__Mul.html#l-onnx-doc-mul
// INPUTS:
//      - A (heterogeneous) - T:  first operand.
//      - B (heterogeneous) - T:  second operand.
// OUTPUTS:
//      - C (heterogeneous) - T:  result.

pub const Mul = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Mul {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        //set the output type:
        if (output_C.ty == tensorZant_lib.TensorType.undefined) output_C.ty = input_A.ty;

        return Mul{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
        };
    }

    pub fn get_output_shape(self: Mul) []usize {
        return self.output_C.getShape();
    }

    pub fn get_input_tensors(self: Mul) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);

        try inputs.append(allocator, self.input_A);
        try inputs.append(allocator, self.input_B);

        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Mul) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);

        try outputs.append(allocator, self.output_C);

        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Mul, writer: *std.Io.Writer) !void {
        //----create tensor_A_string
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);
        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_A.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_A.name) });
        }

        //----create tensor_B_string
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);
        if (self.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.mul_lean({s}, {s}, ({s}), &tensor_{s}) catch return -1;
        , .{
            self.input_A.ty.toString(),
            tensor_A_string, // Input tensor A
            tensor_B_string, // Input tensor B
            try utils.getSanitizedName(self.output_C.name), // Output tensor C
        });
    }

    pub fn compute_output_shape(self: Mul) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_mul_output_shape(self.input_A.shape, self.input_B.shape);
        self.output_C.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Mul) void {
        std.debug.print("\n Mul:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Mul, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_A == old_tensor) {
            self.input_A = new_tensor;
            return;
        }
        if (self.input_B == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.output_C == old_tensor) {
            self.output_C = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn render_lower(self: Mul, builder: *UOpBuilder) !void {
        const A_id = self.input_A.get_tensorZantID();
        const B_id = self.input_B.get_tensorZantID();
        const out_id = self.output_C.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const strideA = self.input_A.stride;
        const strideB = self.input_B.stride;
        const out_dtype = utils.tensorTypeToDtype(self.output_C.ty);

        lowerMul(
            builder,
            A_id,
            B_id,
            out_id,
            out_shape,
            strideA,
            strideB,
            out_dtype,
        );
    }

    /// https://onnx.ai/onnx/operators/onnx__Mul.html
    pub fn lowerMul(
        b: *UOpBuilder,
        A_id: usize, // input-tensor SSA ids
        B_id: usize,
        out_id: usize,
        out_shape: []const usize, // broadcasted shape
        strideA: []const usize, // per-dim strides (0 ⇒ broadcast)
        strideB: []const usize,
        out_dtype: DType, // promoted element type
    ) void { // returns id of result buffer

        // ── Set-up phase ────────────────────────────────────────────────────
        _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)
        _ = b.push(.SHAPE, .i32, &.{B_id}, null); // b_shape  (dbg only)

        const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

        const id_viewB = b.push(.VIEW, out_dtype, &.{B_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideB } });

        // ── Flat element loop ───────────────────────────────────────────────
        var nelem: usize = 1;
        for (out_shape) |d| nelem *= d;

        const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

        const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

        const id_gepB = b.push(.GEP, out_dtype, &.{ id_viewB, id_range }, Any{ .mem_info = .{ .base = id_viewB, .offset = 0, .stride = 1 } });

        const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);
        const id_loadB = b.push(.LOAD, out_dtype, &.{id_gepB}, null);

        const id_mul = b.push(.MUL, out_dtype, &.{ id_loadA, id_loadB }, null);

        const id_gepO = b.push(.GEP, out_dtype, &.{ out_id, id_range }, Any{ .mem_info = .{ .base = out_id, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_mul }, null);

        _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);
    }
};
