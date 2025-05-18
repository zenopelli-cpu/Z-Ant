const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32 {
    const output_2 = try allocator.alloc(f32, 24);
// SHAPE: id=0 src=0 -> shape { 2, 3, 4 }
var idx_3: i32 = 0; // RANGE (uop 3)
while (idx_3 < 24) : (idx_3 += 1) {
        const addr_4 = @intFromPtr(input_0.ptr) + (@as(usize, @intCast(idx_3))) * @sizeOf(f32); // GEP id=4
        var buf_5: f32 = @as(*const f32, @ptrFromInt(addr_4)).*; // LOAD (uop 5)
    _ = &buf_5;
        const addr_6 = @intFromPtr(output_2.ptr) + (@as(usize, @intCast(idx_3))) * @sizeOf(f32); // GEP id=6
        @as(*f32, @ptrFromInt(addr_6)).* = buf_5; // STORE (uop 7)
}
    return output_2;
}
