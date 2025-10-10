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

// https://onnx.ai/onnx/operators/onnx__Pow.html

pub const Pow = struct {
    // inputs
    X: *TensorZant,
    Y: *TensorZant,

    // Zs
    Z: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Pow {
        // Pow has 2-4 inputs
        if (nodeProto.input.len < 2 or nodeProto.input.len > 4) {
            return error.PowInvalidInputCount;
        }

        const X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.X_notFound;
        const Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.Y_notFound;
        const Z = if (tensorZant_lib.tensorMap.getPtr(nodeProto.Z[0])) |ptr| ptr else return error.Z_notFound;

        return Pow{
            .X = X,
            .Y = Y,
            .Z = Z,
        };
    }

    pub fn get__tensors(op: *const Pow) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.Z;
        return tensors;
    }

    pub fn get_input_tensors(op: *const Pow) ![]*TensorZant {
        var count: usize = 2;
        if (op.input_constant_value != null) count += 1;
        if (op.input_axes != null) count += 1;

        var tensors = try allocator.alloc(*TensorZant, count);
        tensors[0] = op.X;
        tensors[1] = op.input_Pows;
        var idx: usize = 2;
        if (op.input_constant_value) |cv| {
            tensors[idx] = cv;
            idx += 1;
        }
        if (op.input_axes) |axes| {
            tensors[idx] = axes;
        }
        return tensors;
    }

    pub fn write_op(op: *const Pow, writer: std.fs.File.Writer) !void {
        // Build input data reference
        var data_ref: []u8 = undefined;
        defer allocator.free(data_ref);
        if (op.X.tc == TensorCategory.INITIALIZER or op.X.tc == TensorCategory.CONSTANT) {
            data_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (op.X.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(op.X.name),
                ")",
            });
        } else {
            data_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(op.X.name),
            });
        }

        // Build Pows reference (always i64 tensor)
        var Pows_ref: []u8 = undefined;
        defer allocator.free(Pows_ref);
        if (op.input_Pows.tc == TensorCategory.INITIALIZER or op.input_Pows.tc == TensorCategory.CONSTANT) {
            Pows_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (op.input_Pows.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(op.input_Pows.name),
                ")",
            });
        } else {
            Pows_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(op.input_Pows.name),
            });
        }

        // Optional constant value
        var const_ref: []const u8 = "null";
        if (op.input_constant_value) |cv| {
            const cv_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (cv.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(cv.name),
                ")",
            });
            const_ref = cv_ref;
            // leak on purpose into generated code scope; freed with allocator lifetime at end of codegen
        }

        // Optional axes
        var axes_ref: []const u8 = "null";
        if (op.input_axes) |ax| {
            const ax_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (ax.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(ax.name),
                ")",
            });
            axes_ref = ax_ref;
        }

        // Z tensor name
        const out_name = try utils.getSanitizedName(op.Z.name);

        // No shape fix needed - the mathematical function will handle shape validation

        // Emit Pow call (preserve dtype)
        _ = try writer.print(
            \\    tensMath.Pow({s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        &tensor_{s},
            \\        "{s}"
            \\    ) catch return -1;
        , .{
            op.Z.ty.toString(),
            data_ref,
            Pows_ref,
            const_ref,
            axes_ref,
            out_name,
            op.mode,
        });
    }

    pub fn print(op: *const Pow) void {
        std.debug.print("\n Pow:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *Pow, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.X == old_tensor) {
            self.X = new_tensor;
            return;
        }
        if (self.input_Pows == old_tensor) {
            self.input_Pows = new_tensor;
            return;
        }
        if (self.input_constant_value != null and self.input_constant_value.? == old_tensor) {
            self.input_constant_value = new_tensor;
            return;
        }
        if (self.input_axes != null and self.input_axes.? == old_tensor) {
            self.input_axes = new_tensor;
            return;
        }
        if (self.Z == old_tensor) {
            self.Z = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
