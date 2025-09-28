const std = @import("std");

// Return codes standardized - matching existing behavior
pub const RC = struct {
    pub const OK: i32 = 0;
    pub const MATH_ERROR: i32 = -1;
    pub const INIT_ERROR: i32 = -2;
    pub const RETURN_ERROR: i32 = -3;
};

// Centralized logging helper
pub fn emitLogHelper(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\
        \\inline fn logMsg(comptime msg: []const u8) void {{
        \\    if (log_function != null) {{
        \\        var buffer: [msg.len + 1:0]u8 = undefined;
        \\        @memcpy(buffer[0..msg.len], msg);
        \\        buffer[msg.len] = 0;
        \\        safe_log_forwarder(@ptrCast(&buffer));
        \\    }}
        \\}}
        \\
        \\inline fn logf(comptime fmt: []const u8, args: anytype) void {{
        \\    if (log_function != null) {{
        \\        var tmp: [256]u8 = undefined;
        \\        const written = std.fmt.bufPrint(&tmp, fmt, args) catch return;
        \\        var zbuf: [257:0]u8 = undefined;
        \\        @memcpy(zbuf[0..written.len], written);
        \\        zbuf[written.len] = 0;
        \\        safe_log_forwarder(@ptrCast(&zbuf));
        \\    }}
        \\}}
        \\
        \\fn logTensorStatsU8(label: []const u8, t: *Tensor(u8)) void {{
        \\    var min_val: u8 = 255;
        \\    var max_val: u8 = 0;
        \\    var sum_val: usize = 0;
        \\    var zeros: usize = 0;
        \\    var i: usize = 0;
        \\    while (i < t.size) : (i += 1) {{
        \\        const v = t.data[i];
        \\        if (v < min_val) min_val = v;
        \\        if (v > max_val) max_val = v;
        \\        sum_val += v;
        \\        if (v == 0) zeros += 1;
        \\    }}
        \\    const mean_val: f32 = @as(f32, @floatFromInt(sum_val)) / @as(f32, @floatFromInt(t.size));
        \\    const s0 = if (t.shape.len > 0) t.shape[0] else 0;
        \\    const s1 = if (t.shape.len > 1) t.shape[1] else 0;
        \\    const s2 = if (t.shape.len > 2) t.shape[2] else 0;
        \\    const s3 = if (t.shape.len > 3) t.shape[3] else 0;
        \\    logf("[conv] {{s}} shape={{{{ {{}} , {{}} , {{}} , {{}} }}}} size={{}} min={{}} max={{}} mean={{}} zeros={{}}\\n", .{{ label, s0, s1, s2, s3, t.size, min_val, max_val, mean_val, zeros }});
        \\}}
        \\
    , .{});
}

// Standard function signature template
pub fn emitFunctionSignature(writer: std.fs.File.Writer, do_export: bool) !void {
    try writer.print(
        \\
        \\ // return codes:
        \\ //  0 : everything good
        \\ // -1 : something went wrong in the mathematical operations
        \\ // -2 : something went wrong in the initialization phase
        \\ // -3 : something went wrong in the output/return phase
        \\pub {s} fn predict (
        \\    input: [*]T_in,
        \\    input_shape: [*]u32,
        \\    shape_len: u32,
        \\    result: *[*]T_out,
        \\) {s} i32 {{
    , .{
        if (do_export) "export" else "",
        if (do_export) "callconv(.C)" else "",
    });
}

// Helper for input size calculation - extracted from common pattern
pub fn emitInputSizeCalculation(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\  
        \\    //computing the size of the input tensor (runtime)
        \\    var input_size: usize = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        input_size *= @as(usize, input_shape[dim_i]);
        \\    }}
    , .{});
}
