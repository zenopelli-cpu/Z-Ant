const std = @import("std");
const zant_allocator = @import("../Utils/allocator.zig");

/// Function pointer type for C weight reading callback
/// Parameters:
/// - offset: Byte offset from the start of weights region
/// - buffer: Destination buffer to read data into
/// - size: Number of bytes to read
/// Returns: 0 on success, non-zero on error
pub const WeightReadCallback = *const fn (offset: usize, buffer: [*]u8, size: usize) callconv(.C) c_int;

/// Global state for weights I/O
var weight_read_callback: ?WeightReadCallback = null;
var weights_base_address: ?[*]const u8 = null;

/// Register a C callback function for reading weights
/// If callback is null, clears the current callback
pub export fn zant_register_weight_callback(callback: ?WeightReadCallback) void {
    weight_read_callback = callback;
}

/// Set the base address for direct weight access (fallback mode)
/// This should point to the start of the weights region in memory
pub export fn zant_set_weights_base_address(base_address: ?[*]const u8) void {
    weights_base_address = base_address;
}

/// Get information about the current weight I/O configuration
pub export fn zant_get_weights_io_info() WeightsIOInfo {
    return WeightsIOInfo{
        .has_callback = weight_read_callback != null,
        .has_base_address = weights_base_address != null,
        .callback_ptr = @intFromPtr(weight_read_callback orelse @as(?WeightReadCallback, null)),
        .base_address = @intFromPtr(weights_base_address orelse @as(?[*]const u8, null)),
    };
}

/// Information structure about weights I/O configuration
pub const WeightsIOInfo = extern struct {
    has_callback: bool,
    has_base_address: bool,
    callback_ptr: usize,
    base_address: usize,
};

/// Read weight data using the configured I/O method
/// This is the main function used by the inference code
/// Returns error if reading fails
pub fn read_weights(comptime T: type, offset: usize, count: usize) ![]const T {
    const byte_size = count * @sizeOf(T);
    const byte_offset = offset;

    if (weight_read_callback) |callback| {
        // Use registered callback
        const buffer = zant_allocator.allocator.alloc(u8, byte_size) catch return error.OutOfMemory;

        const result = callback(byte_offset, buffer.ptr, byte_size);
        if (result != 0) {
            zant_allocator.allocator.free(buffer);
            return error.CallbackFailed;
        }

        // Cast buffer to correct type and return
        const typed_ptr = @as([*]const T, @ptrCast(@alignCast(buffer.ptr)));
        return typed_ptr[0..count];
    } else if (weights_base_address) |base| {
        // Use direct memory access (fallback)
        const byte_ptr = base + byte_offset;
        const typed_ptr = @as([*]const T, @ptrCast(@alignCast(byte_ptr)));
        return typed_ptr[0..count];
    } else {
        // No I/O method configured - this is an error
        return error.NoWeightIOConfigured;
    }
}

/// Read weight data and copy to a provided buffer
/// Useful when the caller wants to manage memory allocation
pub fn read_weights_to_buffer(comptime T: type, offset: usize, buffer: []T) !void {
    const byte_size = buffer.len * @sizeOf(T);
    const byte_offset = offset;

    if (weight_read_callback) |callback| {
        // Use registered callback
        const u8_buffer = @as([*]u8, @ptrCast(buffer.ptr))[0..byte_size];
        const result = callback(byte_offset, u8_buffer.ptr, byte_size);
        if (result != 0) {
            return error.CallbackFailed;
        }
    } else if (weights_base_address) |base| {
        // Use direct memory access (fallback)
        const byte_ptr = base + byte_offset;
        const src_buffer = @as([*]const u8, @ptrCast(byte_ptr))[0..byte_size];
        const dst_buffer = @as([*]u8, @ptrCast(buffer.ptr))[0..byte_size];
        @memcpy(dst_buffer, src_buffer);
    } else {
        return error.NoWeightIOConfigured;
    }
}

/// Get a direct pointer to weight data (only works with direct access mode)
/// This maintains compatibility with existing code that expects direct pointers
pub fn get_weights_ptr(comptime T: type, offset: usize) !*const T {
    if (weights_base_address) |base| {
        const byte_ptr = base + offset;
        return @as(*const T, @ptrCast(@alignCast(byte_ptr)));
    } else {
        return error.DirectAccessNotAvailable;
    }
}

/// Initialize weights I/O with automatic detection of available methods
/// This function should be called during system initialization
pub export fn zant_init_weights_io() void {
    init_weights_io();
}

/// Internal initialization function
pub fn init_weights_io() void {
    // Try to detect the weights base address from linker symbols
    // These symbols should be provided by the linker script
    if (@hasDecl(@This(), "__flash_weights_start__")) {
        const weights_start = @extern([*]const u8, .{ .name = "__flash_weights_start__" });
        weights_base_address = weights_start;
    } else {
        // Fallback: try to use a default XIP address if no linker symbol
        // This can be overridden by calling zant_set_weights_base_address
        weights_base_address = @as([*]const u8, @ptrFromInt(0x08080000)); // Default flash weights address
    }
}

/// For testing and debugging: simulate weight reading
pub fn debug_read_weights(comptime T: type, test_data: []const T, offset: usize, count: usize) []const T {
    const start_idx = offset / @sizeOf(T);
    const end_idx = start_idx + count;
    if (end_idx <= test_data.len) {
        return test_data[start_idx..end_idx];
    } else {
        return test_data[start_idx..];
    }
}

/// Error types for weights I/O operations
pub const WeightIOError = error{
    NoWeightIOConfigured,
    CallbackFailed,
    DirectAccessNotAvailable,
    OutOfMemory,
    InvalidOffset,
    InvalidSize,
};

// Tests
test "weights_io basic functionality" {
    const testing = std.testing;

    // Test with direct access
    const test_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    weights_base_address = @as([*]const u8, @ptrCast(&test_data));

    const result = try read_weights(f32, 0, 3);
    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(f32, 1.0), result[0]);
    try testing.expectEqual(@as(f32, 2.0), result[1]);
    try testing.expectEqual(@as(f32, 3.0), result[2]);

    // Reset state
    weights_base_address = null;
    weight_read_callback = null;
}

test "weights_io callback functionality" {
    const testing = std.testing;

    // Mock callback for testing
    const MockCallback = struct {
        fn mock_read(offset: usize, buffer: [*]u8, size: usize) callconv(.C) c_int {
            _ = offset;
            // Fill buffer with pattern for testing
            var i: usize = 0;
            while (i < size) : (i += 1) {
                buffer[i] = @as(u8, @intCast(i % 256));
            }
            return 0; // Success
        }
    };

    // Register callback
    zant_register_weight_callback(MockCallback.mock_read);

    const result = try read_weights(u8, 0, 4);
    try testing.expectEqual(@as(usize, 4), result.len);
    try testing.expectEqual(@as(u8, 0), result[0]);
    try testing.expectEqual(@as(u8, 1), result[1]);
    try testing.expectEqual(@as(u8, 2), result[2]);
    try testing.expectEqual(@as(u8, 3), result[3]);

    // Clean up
    weight_read_callback = null;
}
