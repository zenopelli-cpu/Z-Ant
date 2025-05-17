const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32 {
    const output_10 = try allocator.alloc(f32, 6);
    const output_2 = try allocator.alloc(f32, 6);
    defer allocator.free(output_2);
    // SHAPE: id=0 src=0 -> shape { 2, 3 }
    const addr_3 = @intFromPtr(input_0.ptr) + (0) * @sizeOf(f32); // GEP id=3
    var buf_4: f32 = @as(*const f32, @ptrFromInt(addr_3)).*; // LOAD (uop 4)
    _ = &buf_4;
    const addr_6 = @intFromPtr(output_2.ptr) + (0) * @sizeOf(f32); // GEP id=6
    @as(*f32, @ptrFromInt(addr_6)).* = buf_4; // STORE (uop 7)
    // SHAPE: id=8 src=2 -> shape { 2, 3 }
    var idx_11: i32 = 0; // RANGE (uop 11)
    while (idx_11 < 6) : (idx_11 += 1) {
        const addr_12 = @intFromPtr(output_2.ptr) + ((((@as(usize, @intCast(idx_11)) % 3) * 1) + (((@as(usize, @intCast(idx_11)) / 3) % 2) * 1))) * @sizeOf(f32); // GEP id=12
        var buf_13: f32 = @as(*const f32, @ptrFromInt(addr_12)).*; // LOAD (uop 13)
        _ = &buf_13;
        const buf_14: f32 = -buf_13; // NEG (uop 14)
        _ = &buf_14;
        const addr_15 = @intFromPtr(output_10.ptr) + (@as(usize, @intCast(idx_11))) * @sizeOf(f32); // GEP id=15
        @as(*f32, @ptrFromInt(addr_15)).* = buf_14; // STORE (uop 16)
    }
    return output_10;
}
