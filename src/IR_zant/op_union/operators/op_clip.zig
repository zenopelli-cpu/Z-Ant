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

// https://onnx.ai/onnx/operators/onnx__Clip.html
// INPUTS:
//      - input (T) T: Input tensor whose elements to be clipped
//      - min (T) T: Minimum value, under which element is replaced by min. Optional, default -inf.
//      - max (T) T: Maximum value, above which element is replaced by max. Optional, default +inf.
// OUTPUTS:
//      - output (T) T: Output tensor with clipped input elements
pub const Clip = struct {
    input: *TensorZant,
    min: ?*TensorZant,
    max: ?*TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Clip {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        // Optional min and max inputs
        const min = if (nodeProto.input.len > 1) tensorZant_lib.tensorMap.getPtr(nodeProto.input[1]) else null;
        const max = if (nodeProto.input.len > 2) tensorZant_lib.tensorMap.getPtr(nodeProto.input[2]) else null;

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input.ty;

        return Clip{
            .input = input,
            .min = min,
            .max = max,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Clip) []usize {
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Clip) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        // Append the mandatory input tensor
        try input_tensors.append(self.input);

        // Append optional min and max tensors if they exist
        if (self.min) |min_tensor| {
            try input_tensors.append(min_tensor);
        }
        if (self.max) |max_tensor| {
            try input_tensors.append(max_tensor);
        }

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Clip) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        // Append the single output tensor
        try output_tensors.append(self.output);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: Clip, writer: std.fs.File.Writer) !void {
        var input_tensor_string: []u8 = undefined;
        var min_tensor_string: []u8 = undefined;
        var max_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);
        errdefer if (self.min != null) allocator.free(min_tensor_string);
        errdefer if (self.max != null) allocator.free(max_tensor_string);

        // Prepare input tensor string
        if (self.input.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try self.input.getNameSanitized(),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try self.input.getNameSanitized() });
        }

        // Prepare min tensor string if it exists
        if (self.min) |min_tensor| {
            if (min_tensor.tc == TensorCategory.INITIALIZER) {
                min_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try min_tensor.getNameSanitized(),
                    ")",
                });
            } else {
                min_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try min_tensor.getNameSanitized() });
            }
        }

        // Prepare max tensor string if it exists
        if (self.max) |max_tensor| {
            if (max_tensor.tc == TensorCategory.INITIALIZER) {
                max_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try max_tensor.getNameSanitized(),
                    ")",
                });
            } else {
                max_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try max_tensor.getNameSanitized() });
            }
        }

        // https://onnx.ai/onnx/operators/onnx__Clip.html
        // pub inline fn lean_clip(
        // comptime T: type,
        // inputTensor: *const Tensor(T),
        // minTensor: ?*const Tensor(T),
        // maxTensor: ?*const Tensor(T),
        // outputTensor: *Tensor(T),
        // ) !void {

        _ = try writer.print(
            \\
            \\
            \\    tensMath.clip_lean(
            \\      {s},  //input type
            \\      {s},  //input tensor
            \\      {s},  //min tensor
            \\      {s},  //max tensor
            \\      &tensor_{s},  //output tensor
            \\    ) catch return -1;
        , .{
            self.input.ty.toString(),
            input_tensor_string,
            if (self.min != null) min_tensor_string else "null",
            if (self.max != null) max_tensor_string else "null",
            try self.output.getNameSanitized(),
        });
    }

    pub fn compute_output_shape(self: Clip) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_clip_output_shape(self.input.get_shape());
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Clip) void {
        std.debug.print("\n Clip:\n {any}", .{self});
    }
};
