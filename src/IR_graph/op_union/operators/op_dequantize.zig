const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const tensorMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensor = @import("../../../Core/Tensor/tensor.zig");
const AnyTensor = tensor.AnyTensor;
const tensorZant = @import("../../tensorZant.zig");
const TensorZant = tensorZant.TensorZant;
const TensorCategory = tensorZant.TensorCategory;
const TensorType = tensorZant.TensorType;
const IR_utils = @import("../../utils.zig"); //this is IR utils
const utils = @import("codegen").utils;

pub const dequantize = struct {
    input: *TensorZant,
    output: *TensorZant,
    //scheme: quantScheme,

    pub fn init(comptime input_type: type, inputTensor: *Tensor(input_type), comptime output_type: type) !dequantize {
        const shape: []usize = inputTensor.shape;

        const input_ptr = try allocator.create(TensorZant);
        input_ptr.* = TensorZant{
            .name = "input",
            .ty = TensorType.fromType(input_type),
            .tc = TensorCategory.INPUT,
            .ptr = AnyTensor{ .input_type = inputTensor },
            .shape = shape,
        };

        const outputTensor = Tensor(input_type).fromShape(&allocator, shape);
        const output_ptr = try allocator.create(TensorZant);
        output_ptr.* = TensorZant{
            .name = "output",
            .ty = TensorType.fromType(output_type),
            .tc = TensorCategory.OUTPUT,
            .ptr = AnyTensor{ .output_type = outputTensor },
            .shape = shape,
        };
        return dequantize{
            .input = input_ptr,
            .output = output_ptr,
        };
    }

    pub fn get_output_shape(self: dequantize) []usize {
        return self.output.getShape();
    }

    pub fn get_output_tensor(self: dequantize) *TensorZant {
        return self.output;
    }

    pub fn compute_output_shape(self: dequantize) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_quantize_output_shape(self.input.getShape());
        return output_shape;
    }

    pub fn write_op(self: dequantize, writer: std.fs.File.Writer) !void {
        const input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(self.input.name),
            ")",
        });

        const output_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(self.output.name),
            ")",
        });

        _ = try writer.print(
            \\
            \\
            \\    tensMath.lean_dequantize(
            \\        {s}, //inputType 
            \\        {s}, //outputType 
            \\        {s}, //input 
            \\        {s}, //output
            \\    )
        , .{
            self.input.ty.toString(),
            self.output.ty.toString(),
            input_tensor_string,
            output_tensor_string,
        });
    }

    pub fn print(self: dequantize) void { // TODO
        std.debug.print("\n dequantize:\n {any}", .{self});
    }
};
