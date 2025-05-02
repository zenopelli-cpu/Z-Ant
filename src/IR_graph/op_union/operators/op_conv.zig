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

pub const Conv = struct {
    input: f32,
    kernel: f16,
    padding: []usize,

    pub fn init(nodeProto: *NodeProto) !Conv {
        _ = nodeProto; //"details" will be a onnx struct
        // Define the padding array
        const padding_array = @constCast(&[_]usize{ 5, 6, 7, 8 });

        // Create a slice from the array
        const padding_slice: []usize = padding_array[0..];

        return Conv{
            .input = 10,
            .kernel = 3.14,
            .padding = padding_slice,
        };
    }

    pub fn get_output_shape() []usize {
        const res: []usize = [_]usize{ 2, 2, 3, 3 };
        return res;
    }

    pub fn print(self: Conv) void {
        std.debug.print("\n CONV:\n {any}", .{self});
    }
};
