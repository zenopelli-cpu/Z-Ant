const std = @import("std");
const allocator = std.heap.page_allocator;

const Sub = struct {
    input: f32,

    pub fn init(details: []const u8) Sub {
        _ = details; //"details" will be a onnx struct
        return Sub{
            .input = 10,
        };
    }

    pub fn get_output_shape() []usize {
        const res: []usize = [_]usize{ 1, 1, 2, 2 };
        return res;
    }

    pub fn print(self: Sub) void {
        std.debug.print("\n SUB:\n {any}", .{self});
    }
};
