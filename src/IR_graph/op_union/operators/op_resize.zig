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

// https://onnx.ai/onnx/operators/onnx__Resize.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
//      - roi (optional) - T2: ROI (region of interest) tensor
//      - scales (optional, heterogeneous) - tensor(float): The scale array along each dimension
//      - sizes (optional, heterogeneous) - tensor(int64): Target size of the output tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Resized output tensor
// ATTRIBUTES:
//      - antialias - INT (default is '0')
//      - axes - INTS
//      - coordinate_transformation_mode - STRING (default is 'half_pixel')
//      - cubic_coeff_a - FLOAT (default is '-0.75')
//      - exclude_outside - INT (default is '0')
//      - extrapolation_value - FLOAT (default is '0.0')
//      - keep_aspect_ratio_policy - STRING (default is 'stretch')
//      - mode - STRING (default is 'nearest')
//      - nearest_mode - STRING (default is 'round_prefer_floor')
//
pub const Resize = struct {
    input_X: *TensorZant,
    input_roi: ?*TensorZant,
    input_scales: ?*TensorZant,
    input_sizes: ?*TensorZant,
    output_Y: *TensorZant,
    // attributes:
    antialias: i64 = 0,
    axes: []i64 = &[_]i64{},
    coordinate_transformation_mode: []const u8 = "half_pixel",
    cubic_coeff_a: f64 = -0.75,
    exclude_outside: i64 = 0,
    extrapolation_value: f64 = 0.0,
    keep_aspect_ratio_policy: []const u8 = "stretch",
    mode: []const u8 = "nearest",
    nearest_mode: []const u8 = "round_prefer_floor",

    pub fn init(nodeProto: *NodeProto) !Resize {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;

        // ---- optional inputs
        const input_roi: ?*TensorZant = if (nodeProto.input.len >= 2) if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_X_notFound else null;
        const input_scales: ?*TensorZant = if (nodeProto.input.len >= 3) if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_roi_notFound else null;
        const input_sizes: ?*TensorZant = if (nodeProto.input.len >= 4) if (tensorZant.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_sizes_notFound else null;

        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        // ---- ATTRIBUTES from NodeProto
        var antialias: i64 = 0;
        var axes: []i64 = &[_]i64{};
        var coordinate_transformation_mode: []const u8 = try allocator.dupe(u8, "half_pixel");
        var cubic_coeff_a: f64 = -0.75;
        var exclude_outside: i64 = 0;
        var extrapolation_value: f64 = 0.0;
        var keep_aspect_ratio_policy: []const u8 = try allocator.dupe(u8, "stretch");
        var mode: []const u8 = try allocator.dupe(u8, "nearest");
        var nearest_mode: []const u8 = try allocator.dupe(u8, "round_prefer_floor");

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "antialias")) |_| {
                if (attr.type == onnx.AttributeType.INT) antialias = attr.i else return error.ResizeAnitialiasNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "axes")) |_| {
                if (attr.type == onnx.AttributeType.INTS) axes = attr.ints else return error.ResizeAxesNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "coordinate_transformation_mode")) |_| {
                if (attr.type == onnx.AttributeType.STRING) coordinate_transformation_mode = attr.s else return error.Resize_coordinate_transformation_mode_NotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "cubic_coeff_a")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) cubic_coeff_a = attr.f else return error.Resize_cubic_coeff_a_NotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "exclude_outside")) |_| {
                if (attr.type == onnx.AttributeType.INT) exclude_outside = attr.i else return error.Resize_exclude_outside_NotINT;
            } else if (std.mem.indexOf(u8, attr.name, "extrapolation_value")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) extrapolation_value = attr.f else return error.Resize_extrapolation_value_NotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "keep_aspect_ratio_policy")) |_| {
                if (attr.type == onnx.AttributeType.STRING) keep_aspect_ratio_policy = attr.s else return error.Resize_keep_aspect_ratio_policy_NotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "mode")) |_| {
                if (attr.type == onnx.AttributeType.STRING) mode = attr.s else return error.Resize_mode_NotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "nearest_mode")) |_| {
                if (attr.type == onnx.AttributeType.STRING) nearest_mode = attr.s else return error.Resize_nearest_mode_NotSTRING;
            }
        }

        return Resize{
            .input_X = input_X,
            .input_roi = input_roi,
            .mode = mode,
            .input_scales = input_scales,
            .input_sizes = input_sizes,
            .coordinate_transformation_mode = coordinate_transformation_mode,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Resize) []usize {
        return self.output_Y.shape;
    }

    pub fn print(self: Resize) void {
        std.debug.print("\n Resize :{any}\n", .{self});
    }
};
