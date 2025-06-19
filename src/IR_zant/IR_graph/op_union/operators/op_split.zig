const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.IR_graph.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.IR_codegen.utils;

//https://onnx.ai/onnx/operators/onnx__Split.html
// INPUTS:
//      - input (heterogeneous) - T: Input tensor
//      - split (optional, heterogeneous) - tensor(int64):
// OUTPUTS:
//      - output (heterogeneous) - T: Output tensor
// ATTRIBUTES:
//      - axis - INT (default is '0'): Indicate up to which input dimension should be split.
//      - num_outputs - INT: Number of outputs
pub const Split = struct {
    input: *TensorZant,
    split: ?*TensorZant,
    output_Y: *TensorZant,
    //attributes:
    axis: i64 = 0, // default = 0,
    num_outputs: i64 = undefined,

    pub fn init(nodeProto: *NodeProto) !Split {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const splitTensor = if (nodeProto.input.len > 1) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.axes_notFound else null;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = 0;
        var num_outputs: i64 = undefined;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i;
            }
            if (std.mem.eql(u8, attr.name, "num_outputs")) {
                if (attr.type == onnx.AttributeType.INT) num_outputs = attr.i;
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input.ty;

        return Split{
            .input = input,
            .split = splitTensor,
            .output_Y = output_Y,
            .axis = axis,
            .num_outputs = num_outputs,
        };
    }

    pub fn get_output_shape(self: Split) []usize {
        return self.output_Y.shape;
    }

    pub fn get_input_tensors(self: Split) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input);
        if (self.split) |s| try inputs.append(s);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Split) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape() ![]usize {} // TODO

    pub fn print(self: Split) void {
        std.debug.print("\n Split: {any}", .{self});
    }

    pub fn write_op(self: Split, writer: std.fs.File.Writer) !void {
        // --- Crea stringa per input
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);

        if (self.input.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input.name),
            });
        }

        // --- Crea stringa per split (se presente)
        var tensor_split_string: []u8 = undefined;
        defer allocator.free(tensor_split_string);

        if (self.split != null) {
            if (self.split.?.tc == TensorCategory.INITIALIZER) {
                tensor_split_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(self.split.?.name),
                    ")",
                });
            } else {
                tensor_split_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "&tensor_",
                    try utils.getSanitizedName(self.split.?.name),
                });
            }
        }

        _ = try writer.print(
            \\    tensMath.split_lean(
            \\        {s},
            \\        {s}, // input tensor
            \\        {s}, // split tensor
            \\        &tensor_{s}, // output Y
            \\        {d}, // axis
            \\        {d}, // num_outputs
            \\    );
        , .{
            self.input.ty.toString(),
            tensor_input_string,
            tensor_split_string,
            try utils.getSanitizedName(self.output_Y.name),
            self.axis,
            self.num_outputs,
        });
    }
};
