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

/// Fused Conv+Clip operation for better performance
/// This combines convolution followed by clipping (typically ReLU6: clip(x, 0, 6))
/// Common pattern in MobileNet v2 and other efficient architectures
pub const ConvClip = struct {
    // Conv inputs
    input_X: *TensorZant,
    input_W: *TensorZant,
    input_B: ?*TensorZant,

    // Clip parameters
    min: ?*TensorZant,
    max: ?*TensorZant,

    // Output
    output_Y: *TensorZant,

    // Conv attributes
    auto_pad: []const u8,
    dilations: ?[]i64,
    group: i64,
    kernel_shape: ?[]i64,
    pads: ?[]i64,
    strides: ?[]i64,

    pub fn init_from_conv_clip(conv_node: NodeProto, clip_node: NodeProto) !ConvClip {
        // Get Conv inputs
        const input_X = if (tensorZant_lib.tensorMap.getPtr(conv_node.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_W = if (tensorZant_lib.tensorMap.getPtr(conv_node.input[1])) |ptr| ptr else return error.input_W_notFound;
        const input_B = if (conv_node.input.len > 2) if (tensorZant_lib.tensorMap.getPtr(conv_node.input[2])) |ptr| ptr else return error.input_B_notFound else null;

        // Get Clip parameters
        const min = if (clip_node.input.len > 1) tensorZant_lib.tensorMap.getPtr(clip_node.input[1]) else null;
        const max = if (clip_node.input.len > 2) tensorZant_lib.tensorMap.getPtr(clip_node.input[2]) else null;

        // Output is the clip output
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(clip_node.output[0])) |ptr| ptr else return error.output_Y_notFound;

        // Parse Conv attributes
        var auto_pad: []const u8 = "NOTSET";
        var dilations: ?[]i64 = null;
        var group: i64 = 1;
        var kernel_shape: ?[]i64 = null;
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null;

        for (conv_node.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "auto_pad")) {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s;
            }
            if (std.mem.eql(u8, attr.name, "dilations")) {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints;
            }
            if (std.mem.eql(u8, attr.name, "group")) {
                if (attr.type == onnx.AttributeType.INT) group = attr.i;
            }
            if (std.mem.eql(u8, attr.name, "kernel_shape")) {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints;
            }
            if (std.mem.eql(u8, attr.name, "pads")) {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints;
            }
            if (std.mem.eql(u8, attr.name, "strides")) {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints;
            }
        }

        // Set output type
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return ConvClip{
            .input_X = input_X,
            .input_W = input_W,
            .input_B = input_B,
            .min = min,
            .max = max,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .dilations = dilations,
            .group = group,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: ConvClip) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: ConvClip) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_X);
        try inputs.append(self.input_W);
        if (self.input_B) |bias| try inputs.append(bias);
        if (self.min) |min_tensor| try inputs.append(min_tensor);
        if (self.max) |max_tensor| try inputs.append(max_tensor);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: ConvClip) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: ConvClip, writer: std.fs.File.Writer) !void {
        // Build Conv operation strings (similar to op_conv.zig)
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        }

        var tensor_W_string: []u8 = undefined;
        defer allocator.free(tensor_W_string);
        if (self.input_W.tc == TensorCategory.INITIALIZER) {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_W.name),
                ")",
            });
        } else {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_W.name), ")" });
        }

        var bias_string: []u8 = undefined;
        defer allocator.free(bias_string);
        if (self.input_B) |input_B| {
            const B_name = try utils.getSanitizedName(input_B.name);
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
        } else {
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        // Build stride string
        if (self.strides == null) return error.StrideNotFound;
        const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(self.strides.?);

        // Build pads string
        var pads_string: []const u8 = "null";
        if (self.pads != null) {
            if (self.pads.?.len > 0) {
                pads_string = try utils.i64SliceToUsizeArrayString(self.pads.?);
            } else {
                pads_string = "&[_]usize{}";
            }
        }

        // Build dilations string
        var dilat_string: []const u8 = "null";
        if (self.dilations != null) {
            if (self.dilations.?.len > 0) {
                dilat_string = try utils.i64SliceToUsizeArrayString(self.dilations.?);
            } else {
                dilat_string = "&[_]usize{1} ** 2";
            }
        }

        // Build clip bounds strings
        var min_string: []const u8 = "null";
        var max_string: []const u8 = "null";

        if (self.min) |min_tensor| {
            if (min_tensor.tc == TensorCategory.INITIALIZER) {
                min_string = try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{try utils.getSanitizedName(min_tensor.name)});
            } else {
                min_string = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{try utils.getSanitizedName(min_tensor.name)});
            }
        }
        defer if (!std.mem.eql(u8, min_string, "null")) allocator.free(@constCast(min_string));

        if (self.max) |max_tensor| {
            if (max_tensor.tc == TensorCategory.INITIALIZER) {
                max_string = try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{try utils.getSanitizedName(max_tensor.name)});
            } else {
                max_string = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{try utils.getSanitizedName(max_tensor.name)});
            }
        }
        defer if (!std.mem.eql(u8, max_string, "null")) allocator.free(@constCast(max_string));

        // Handle mixed precision (similar to op_conv.zig)
        const target_type = self.output_Y.ty.toString();
        const need_kernel_cast = !std.mem.eql(u8, self.input_W.ty.toString(), target_type);
        const need_bias_cast = if (self.input_B) |bias| !std.mem.eql(u8, bias.ty.toString(), target_type) else false;

        var final_kernel_string: []const u8 = tensor_W_string;
        var final_bias_string: []const u8 = bias_string;
        var need_free_kernel = false;
        var need_free_bias = false;
        defer if (need_free_kernel) allocator.free(@constCast(final_kernel_string));
        defer if (need_free_bias) allocator.free(@constCast(final_bias_string));

        const output_name_sanitized = try utils.getSanitizedName(self.output_Y.name);

        if (need_kernel_cast) {
            final_kernel_string = try std.fmt.allocPrint(allocator, "&tensor_{s}_W_casted_{s}", .{ try utils.getSanitizedName(self.input_W.name), output_name_sanitized });
            need_free_kernel = true;

            _ = try writer.print(
                \\    var tensor_{s}_W_casted_{s} = Tensor({s}).fromShape(&allocator, &tensor_{s}.shape) catch return -2;
                \\    tensMath.cast_lean({s}, &tensor_{s}, &tensor_{s}_W_casted_{s}) catch return -1;
                \\
            , .{
                try utils.getSanitizedName(self.input_W.name), output_name_sanitized,      target_type,
                try utils.getSanitizedName(self.input_W.name), self.input_W.ty.toString(), try utils.getSanitizedName(self.input_W.name),
                try utils.getSanitizedName(self.input_W.name), output_name_sanitized,
            });
        }

        if (need_bias_cast and self.input_B != null) {
            final_bias_string = try std.fmt.allocPrint(allocator, "&tensor_{s}_B_casted_{s}", .{ try utils.getSanitizedName(self.input_B.?.name), output_name_sanitized });
            need_free_bias = true;

            _ = try writer.print(
                \\    var tensor_{s}_B_casted_{s} = Tensor({s}).fromShape(&allocator, &tensor_{s}.shape) catch return -2;
                \\    tensMath.cast_lean({s}, &tensor_{s}, &tensor_{s}_B_casted_{s}) catch return -1;
                \\
            , .{
                try utils.getSanitizedName(self.input_B.?.name), output_name_sanitized,        target_type,
                try utils.getSanitizedName(self.input_B.?.name), self.input_B.?.ty.toString(), try utils.getSanitizedName(self.input_B.?.name),
                try utils.getSanitizedName(self.input_B.?.name), output_name_sanitized,
            });
        }

        // Generate the fused conv+clip call
        _ = try writer.print(
            \\    
            \\    @setEvalBranchQuota(10000);
            \\    // Fused Conv+Clip operation for better performance
            \\    tensMath.conv_clip_lean(
            \\        {s}, //type
            \\        {s}, //input
            \\        {s}, //kernel
            \\        &tensor_{s}, //output
            \\        {s}, //bias
            \\        {s}, //stride
            \\        {s}, //pads
            \\        {s}, //dilations
            \\        {}, //group
            \\        "{s}", //auto_pad
            \\        {s}, //min
            \\        {s}, //max
            \\    ) catch return -1;
        , .{
            target_type,
            tensor_X_string, //Input
            final_kernel_string, //Kernel (possibly casted)
            output_name_sanitized, //Output
            final_bias_string, //Bias (possibly casted)
            stride_string, //Strides
            pads_string, //Pads
            dilat_string, //Dilations
            self.group, //Group
            self.auto_pad, //auto_pad
            min_string, //min
            max_string, //max
        });
    }

    pub fn compute_output_shape(self: ConvClip) []usize {
        var output_shape: []usize = undefined;
        const input_shape = self.input_X.get_shape();
        const kernel_shape = self.input_W.get_shape();
        const stride = self.strides;
        const pads = self.pads;
        const dilations = self.dilations;
        const auto_pad = self.auto_pad;
        output_shape = try tensorMath.get_convolution_output_shape(
            input_shape,
            kernel_shape,
            try utils.i64SliceToUsizeSlice(stride.?),
            if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
            try utils.i64SliceToUsizeSlice(dilations.?),
            auto_pad,
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: ConvClip) void {
        std.debug.print("\n CONV+CLIP FUSED:\n {any}", .{self});
    }
};
