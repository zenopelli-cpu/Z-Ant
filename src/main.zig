const std = @import("std");
const model_opts = @import("model_opts"); // Importa il modulo aggiunto in build.zig

// Declare the external C ABI function exported by the static library
// Assumes T = f32 based on build options
extern fn predict(
    input: [*]const f32, // Changed to const T
    input_shape: [*]const u32, // Changed to const u32
    shape_len: u32,
    result: *[*]f32, // Pointer to receive the output slice pointer
) void;

fn prepareInputData(allocator: std.mem.Allocator) ![]model_opts.data_type {
    const shape = model_opts.input_shape;
    var total_size: usize = 1;
    for (shape) |dim| {
        total_size *= dim;
    }

    const data = try allocator.alloc(model_opts.data_type, total_size);
    errdefer allocator.free(data);

    for (data, 0..) |*val, i| {
        val.* = @as(model_opts.data_type, @floatFromInt(i));
    }

    return data;
}

fn getPredictOutputSize() usize {
    return 1 * 84 * 1344;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Preparing input data...\\n", .{});
    const input_data = try prepareInputData(allocator);
    const input_shape = model_opts.input_shape;

    var output_ptr: [*]model_opts.data_type = undefined;

    std.debug.print("Calling predict (via model_opts.lib)...\\n", .{});

    model_opts.lib.predict(
        input_data.ptr,
        @constCast(@ptrCast(&input_shape)),
        @intCast(input_shape.len),
        &output_ptr,
    );

    std.debug.print("Predict call finished.\n", .{});

    const output_size = getPredictOutputSize();

    // Check if output_ptr is null before creating slice
    // Use @intFromPtr to check if the pointer address is 0 (NULL)
    if (@intFromPtr(output_ptr) == 0) {
        std.debug.print("Error: predict returned a null pointer.\n", .{});
        allocator.free(input_data);
        return;
    }

    const output_slice = @as([*]model_opts.data_type, @ptrCast(output_ptr))[0..output_size];

    //print the output
    std.debug.print("Output (first 10 elements):\n", .{});
    var i: usize = 0;
    while (i < output_slice.len and i < 10) : (i += 1) {
        std.debug.print("{d}, ", .{output_slice[i]});
    }
    if (output_slice.len > 10) {
        std.debug.print("...\n", .{});
    } else {
        std.debug.print("\n", .{});
    }

    std.debug.print("WARNING: Memory for the predict output was NOT freed!\n", .{});

    std.debug.print("Attempting to free input memory...\\n", .{});
    allocator.free(input_data);

    std.debug.print("Program finished.\\n", .{});
}
