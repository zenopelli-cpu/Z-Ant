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

// https://onnx.ai/onnx/operators/onnx__Relu.html#l-onnx-doc-relu
// INPUTS:
//      - X (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - Y (heterogeneous) - T:  output tensor.

pub const Relu = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Relu {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Relu{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Relu) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Relu) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);

        try inputs.append(allocator, self.input_X);

        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Relu) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);

        try outputs.append(allocator, self.output_Y);

        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Relu, writer: *std.Io.Writer) !void {
        //----create tensor_A_string
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);
        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.ReLU_lean({s}, {s}, &tensor_{s}) catch return -1;
        , .{
            self.output_Y.ty.toString(),
            tensor_A_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn compute_output_shape(self: Relu) []usize {
        var output_shape: []usize = undefined;
        output_shape = self.input_X.shape;
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Relu) void {
        std.debug.print("\n Relu:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Relu, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn render_lower(self: Relu, builder: *UOpBuilder) !void {
        const X_id = self.input_X.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.output_Y.ty);

        const out_buf_id = lowerReLU(
            builder,
            X_id,
            out_shape,
            out_dtype,
        );
        _ = out_buf_id;
    }

    /// https://onnx.ai/onnx/operators/onnx__Relu.html
    pub fn lowerReLU(
        b: *UOpBuilder,
        X_id: usize, // SSA id of input tensor X
        x_shape: []const usize, // Shape of input tensor X
        out_dtype: DType,
    ) usize {

        // ── Tiny helpers to reduce boilerplate ────────────────────────────
        const r = struct {
            fn rng(bi: *UOpBuilder, end: usize) usize { // RANGE 0..end-1
                return bi.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = end } });
            }
            fn kconst(bi: *UOpBuilder, v: f32, odtype: DType) usize { // CONST <v> (float)
                return bi.push(.CONST, odtype, &.{}, Any{ .float = v });
            }
        };

        // ── 1. Create a logical view for the input tensor ─────────────────
        const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{ .view_meta = .{ .shape = x_shape, .strides = &.{1} } });

        // ── 2. Create output tensor with the same shape as input ──────────
        const id_Y = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = x_shape });

        // ── 3. Create constants for the operation ─────────────────────────
        const id_zero = r.kconst(b, 0.0, out_dtype); // Zero for comparison

        // ── 4. Create nested loops for each dimension of the tensor ───────
        var nelem: usize = 1;
        for (x_shape) |d| nelem *= d;

        const id_range = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

        // ── 5. Create GEP operation for current element ───────────────────
        const id_gepX = b.push(.GEP, out_dtype, &.{ id_viewX, id_range }, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });

        // ── 6. Load the input value ─────────────────────────────────────────
        const id_x = b.push(.LOAD, out_dtype, &.{id_gepX}, null);

        // ── 7. Implement ReLU: f(x) = 0 if x < 0 else x ────────
        // Compare x with zero
        const id_lt = b.push(.CMPLT, .bool, &.{ id_x, id_zero }, null);

        // Select between x and alpha*x based on comparison result
        const id_result = b.push(.WHERE, out_dtype, &.{ id_lt, id_zero, id_x }, null);

        // ── 8. Store the result to the output tensor ───────────────────────
        const id_gepY = b.push(.GEP, out_dtype, &.{ id_Y, id_range }, Any{ .mem_info = .{ .base = id_Y, .offset = 0, .stride = 1 } });
        _ = b.push(.STORE, out_dtype, &.{ id_gepY, id_result }, null);

        // ── 9. Close all the nested loops ───────────────────────────────────
        _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

        return id_Y; // SSA id of the produced output tensor Y
    }
};
