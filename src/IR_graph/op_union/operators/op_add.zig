const std = @import("std");
const allocator = std.heap.page_allocator;

// https://onnx.ai/onnx/operators/onnx__Add.html
// INPUTS:
//      - A (heterogeneous) - T: First operand.
//      - B (heterogeneous) - T: Second operand.
// OUTPUTS:
//      - C (heterogeneous) - T: Result, has same element type as two inputs.

const Add = struct {
    input: f32,

    pub fn init(details: []const u8) Add {
        _ = details; //"details" will be a onnx struct
        return Add{
            .input = 90,
        };
    }

    pub fn get_output_shape(self: Add) []usize {
        const res: []usize = [_]usize{ 0, 0, 1, 1 };
        res[0] += self.input;
        return res;
    }

    pub fn print(self: Add) void {
        std.debug.print("\n ADD:\n {any}", .{self});
    }
};
