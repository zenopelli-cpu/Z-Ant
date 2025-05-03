const std = @import("std");
const zant = @import("../../../zant.zig");
const allocator = std.heap.page_allocator;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;

// https://onnx.ai/onnx/operators/onnx__Conv.html
// INPUTS:
//      - X (heterogeneous) - T: Input data tensor
//      - W (heterogeneous) - T: The weight tensor
//      - B (optional, heterogeneous) - T: Optional 1D bias to be added to the convolution, has size of M.
// OUTPUTS:
//      - Y (heterogeneous) - T: Output data tensor that contains the result of the convolution
// ATTRIBUTES:
//      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET
//      - dilations - INTS : dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
//      - group - INT (default is '1'): number of groups input channels and output channels are divided into
//      - kernel_shape - INTS : The shape of the convolution kernel. If not present, should be inferred from input W
//      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
//      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

pub const Conv = struct {
    input_X: *TensorZant,
    input_W: *TensorZant,
    input_B: ?*TensorZant,
    output_Y: *TensorZant,
    //attributes:
    auto_pad: []const u8,
    dilations: ?[]i64,
    group: i64,
    kernel_shape: ?[]i64,
    pads: ?[]i64,
    strides: ?[]i64,

    pub fn init(nodeProto: *NodeProto) !Conv {
        const input_X = if (tensorZant.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_W = if (tensorZant.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_W_notFound;
        const input_B = if (nodeProto.input.len > 2) if (tensorZant.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_B_notFound else null;
        const output_Y = if (tensorZant.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var auto_pad: []const u8 = "NOTSET";
        var dilations: ?[]i64 = null;
        var group: i64 = 1;
        var kernel_shape: ?[]i64 = null;
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null; //mandatory

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
                if (attr.type == onnx.AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
            }
        }

        return Conv{
            .input_X = input_X,
            .input_W = input_W,
            .input_B = input_B,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .dilations = dilations,
            .group = group,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };
    }

    pub fn get_output_shape() []usize { //TODO
        const res: []usize = [_]usize{ 2, 2, 3, 3 };
        return res;
    }

    pub fn print(self: Conv) void { //TODO
        std.debug.print("\n CONV:\n {any}", .{self});
    }
};
