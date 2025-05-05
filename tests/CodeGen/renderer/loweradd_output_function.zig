pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32, input_1: []const f32) ![]f32 {
    const buf_7 = try allocator.alloc(f32, 1);
    defer allocator.free(buf_7);
    const buf_6 = try allocator.alloc(f32, 1);
    defer allocator.free(buf_6);
    const buf_8 = try allocator.alloc(f32, 1);
    defer allocator.free(buf_8);
    const output_2 = try allocator.alloc(f32, 6);
var idx_3: i32 = 0; // RANGE (uop 3)
while (idx_3 < 6) : (idx_3 += 1) {
    const addr_4 = @intFromPtr(input_0.ptr) + ((0) + (@as(usize, @intCast(idx_3))) * 1) * @sizeOf(f32); // GEP (uop 4)
    const addr_5 = @intFromPtr(input_1.ptr) + ((0) + (@as(usize, @intCast(idx_3))) * 1) * @sizeOf(f32); // GEP (uop 5)
    buf_6[0] = @as(*const f32, @ptrFromInt(addr_4)).*; // LOAD (uop 6)
    buf_7[0] = @as(*const f32, @ptrFromInt(addr_5)).*; // LOAD (uop 7)
    buf_8[0] = buf_6[0] + buf_7[0]; // ADD (uop 8)
    const addr_9 = @intFromPtr(output_2.ptr) + ((0) + (@as(usize, @intCast(idx_3))) * 1) * @sizeOf(f32); // GEP (uop 9)
    @as(*f32, @ptrFromInt(addr_9)).* = buf_8[0]; // STORE (uop 10)
    }
    return output_2;
}
