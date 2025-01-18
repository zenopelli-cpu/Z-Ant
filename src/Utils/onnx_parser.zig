const std = @import("std");

// function to parse .ONNX files
pub fn readONNXFile(allocator: *std.mem.Allocator, file_path: []const u8) ![]u8 {
    // Ensure the file has an .onnx extension
    if (!file_path.endsWithSlice(".onnx")) {
        return error.InvalidExtension;
    }

    // Open the ONNX file in read mode
    const file = try std.fs.cwd().openFile(file_path, .{ .read = true });
    defer file.close();

    // Get the file size
    const file_size = try file.getEndPos();

    // Allocate a buffer to hold the file contents
    const buffer = try allocator.alloc(u8, file_size);
    defer allocator.free(buffer);

    // Read the file contents into the buffer
    try file.readAll(buffer);

    // Return the buffer containing the file's content
    return buffer;
}
