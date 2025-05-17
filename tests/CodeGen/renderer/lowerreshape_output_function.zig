const std = @import("std");

pub fn generated_kernel(allocator: std.mem.Allocator, input_0: []const f32) ![]f32 {
    const output_1 = try allocator.alloc(f32, 6);
// SHAPE: id=0 src=0 -> shape {  }
    const addr_2 = @intFromPtr(input_0.ptr) + (0) * @sizeOf(f32); // GEP id=2
    var buf_3: f32 = @as(*const f32, @ptrFromInt(addr_2)).*; // LOAD (uop 3)
    _ = &buf_3;
    const addr_4 = @intFromPtr(output_1.ptr) + (0) * @sizeOf(f32); // GEP id=4
    @as(*f32, @ptrFromInt(addr_4)).* = buf_3; // STORE (uop 5)
    const addr_6 = @intFromPtr(input_0.ptr) + (1) * @sizeOf(f32); // GEP id=6
    var buf_7: f32 = @as(*const f32, @ptrFromInt(addr_6)).*; // LOAD (uop 7)
    _ = &buf_7;
    const addr_8 = @intFromPtr(output_1.ptr) + (1) * @sizeOf(f32); // GEP id=8
    @as(*f32, @ptrFromInt(addr_8)).* = buf_7; // STORE (uop 9)
    const addr_10 = @intFromPtr(input_0.ptr) + (2) * @sizeOf(f32); // GEP id=10
    var buf_11: f32 = @as(*const f32, @ptrFromInt(addr_10)).*; // LOAD (uop 11)
    _ = &buf_11;
    const addr_12 = @intFromPtr(output_1.ptr) + (2) * @sizeOf(f32); // GEP id=12
    @as(*f32, @ptrFromInt(addr_12)).* = buf_11; // STORE (uop 13)
    const addr_14 = @intFromPtr(input_0.ptr) + (3) * @sizeOf(f32); // GEP id=14
    var buf_15: f32 = @as(*const f32, @ptrFromInt(addr_14)).*; // LOAD (uop 15)
    _ = &buf_15;
    const addr_16 = @intFromPtr(output_1.ptr) + (3) * @sizeOf(f32); // GEP id=16
    @as(*f32, @ptrFromInt(addr_16)).* = buf_15; // STORE (uop 17)
    const addr_18 = @intFromPtr(input_0.ptr) + (4) * @sizeOf(f32); // GEP id=18
    var buf_19: f32 = @as(*const f32, @ptrFromInt(addr_18)).*; // LOAD (uop 19)
    _ = &buf_19;
    const addr_20 = @intFromPtr(output_1.ptr) + (4) * @sizeOf(f32); // GEP id=20
    @as(*f32, @ptrFromInt(addr_20)).* = buf_19; // STORE (uop 21)
    const addr_22 = @intFromPtr(input_0.ptr) + (5) * @sizeOf(f32); // GEP id=22
    var buf_23: f32 = @as(*const f32, @ptrFromInt(addr_22)).*; // LOAD (uop 23)
    _ = &buf_23;
    const addr_24 = @intFromPtr(output_1.ptr) + (5) * @sizeOf(f32); // GEP id=24
    @as(*f32, @ptrFromInt(addr_24)).* = buf_23; // STORE (uop 25)
    return output_1;
}
