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
const tensorMath = zant.core.tensor.math_standard;
const utils = @import("../../../CodeGen/utils.zig");

//https://onnx.ai/onnx/operators/onnx__MaxPool.html
// INPUTS:
//      - X (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - Y (heterogeneous) - T:  output tensor.
//      - indices (optional, heterogeneous) - T:  output indices tensor.
// ATTRIBUTES:
//      - auto_pad (string) - AutoPad type. Default is NOTSET.
//      - ceil_mode (int) - Ceil mode. Default is 0.
//      - dilations (list of ints) - Dilation value. Default is 1.
//      - kernel_shape (list of ints) - Kernel shape.
//      - pads (list of ints) - Padding value. Default is 0.
//      - storage_order (int) - Storage order. Default is 0.
//      - strides (list of ints) - Stride value. Default is 1.

pub const MaxPool = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    output_indices: ?*TensorZant,
    //attributes:
    auto_pad: []const u8, // default = "NOTSET",
    ceil_mode: i64, // default = 0;
    dilations: ?[]i64, // default = null;
    kernel_shape: ?[]i64, // default = null; but mandatory
    pads: ?[]i64, // default = null;
    storage_order: i64, // default = 0;
    strides: ?[]i64, // default = null;

    pub fn init(nodeProto: *NodeProto) !MaxPool {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;
        const output_indices = if (nodeProto.output.len > 1) if (tensorZant.tensorMap.getPtr(nodeProto.output[1])) |ptr| ptr else return error.output_indices_notFound else null;

        var auto_pad: []const u8 = "NOTSET";
        var ceil_mode: i64 = 0;
        var dilations: ?[]i64 = null;
        var kernel_shape: ?[]i64 = null; //mandatory
        var pads: ?[]i64 = null;
        var storage_order: i64 = 0;
        var strides: ?[]i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.MaxPoolAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
                if (attr.type == onnx.AttributeType.INT) ceil_mode = attr.i else return error.MaxPoolCeil_modeNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.MaxPoolDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.MaxPoolKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.MaxPoolPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "storage_order")) |_| {
                if (attr.type == onnx.AttributeType.INT) storage_order = attr.i else return error.MaxPoolStorage_orderNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.MaxPoolStridesNotINTS;
            }
        }
        return MaxPool{
            .input_X = input_X,
            .output_Y = output_Y,
            .output_indices = output_indices,
            .auto_pad = auto_pad,
            .ceil_mode = ceil_mode,
            .dilations = dilations,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .storage_order = storage_order,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: MaxPool) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input_X;
        return res;
    }

    pub fn compute_output_shape(self: MaxPool) []usize {
        var output_shape: []usize = undefined;
        const kernel_shape = self.kernel_shape;
        const strides = self.strides;
        output_shape = try tensorMath.get_pooling_output_shape(
            self.input_X.shape,
            try utils.i64SliceToUsizeSlice(kernel_shape.?),
            try utils.i64SliceToUsizeSlice(strides.?),
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: MaxPool) void { // TODO
        std.debug.print("\n AveragePool:\n {any}", .{self});
    }
};
