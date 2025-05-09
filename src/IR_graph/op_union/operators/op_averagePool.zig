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
//      - X (heterogeneous) - T: Input data tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output data tensor from average or max pooling across the input tensor.
//      - (NOT IMPLEMENTED) Indices (optional, heterogeneous) - I: Indices tensor from max pooling across the input tensor.
// ATTRIBUTES:
//      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
//      - ceil_mode - INT (default is '0'): Whether to use ceil or floor (default) to compute the output shape
//      - dilations - INTS : Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis
//      - kernel_shape - INTS (required) : The size of the kernel along each axis.
//      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
//      - storage_order - INT (default is '0'): The storage order of the tensor. 0 is row major, and 1 is column major. This attribute is used only to convert an n-tuple index value into a single integer value for producing the second output.
//      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

pub const AveragePool = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    //attributes:
    auto_pad: []const u8, // default = "NOTSET",
    ceil_mode: i64, // default = 0;
    dilations: ?[]i64, // default = null;
    kernel_shape: ?[]i64, // default = null; but mandatory
    pads: ?[]i64, // default = null;
    storage_order: i64, // default = 0;
    strides: ?[]i64, // default = null;

    pub fn init(nodeProto: *NodeProto) !AveragePool {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

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

        return AveragePool{
            .input_X = input_X,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .ceil_mode = ceil_mode,
            .dilations = dilations,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .storage_order = storage_order,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: AveragePool) []usize { // TODO
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn compute_averagePool_output_shape(self: AveragePool) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_onnx_averagepool_output_shape(
            self.input_X.shape,
            try utils.i64SliceToUsizeSlice(self.kernel_shape.?),
            try utils.i64SliceToUsizeSlice(self.strides.?),
            try utils.i64SliceToUsizeSlice(self.dilations.?),
            try utils.i64SliceToUsizeSlice(self.pads.?),
            "NOTSET",
            try utils.i64SliceToUsizeSlice(self.ceil_mode),
        );
        return output_shape;
    }

    pub fn print(self: AveragePool) void { // TODO
        std.debug.print("\n AveragePool:\n {any}", .{self});
    }
};
