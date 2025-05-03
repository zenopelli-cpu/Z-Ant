const std = @import("std");
const testing = std.testing;
const jpegParser = @import("/workspaces/Z-Ant/src/ImageToTensor/jpeg/jpegParser.zig");
const decoder = @import("/workspaces/Z-Ant/src/ImageToTensor/utils.zig");
const formatVerifier = @import("/workspaces/Z-Ant/src/ImageToTensor/formatVerifier.zig");
const jpegToYCbCr = @import("/workspaces/Z-Ant/src/ImageToTensor/jpeg/jpegDecoder.zig");
const jpegAlgorithms = @import("/workspaces/Z-Ant/src/ImageToTensor/jpeg/jpegAlgorithms.zig");
const writerBMP = @import("/workspaces/Z-Ant/src/ImageToTensor/writerBMP.zig");

test "JPEG parsing" {
    // Test parsing a valid JPEG file
    const test_file_path = "/home/alessandro/Desktop/jpeg/tests/gorilla.jpg";
    var file = try std.fs.cwd().openFile(test_file_path, .{});
    defer file.close();

    var buffer: [5 * 1024 * 1024]u8 = undefined; // 1MB buffer
    const bytes_read = try file.readAll(&buffer);

    var jpeg_data = try jpegParser.jpegParser(testing.allocator, buffer[0..bytes_read]);
    defer jpeg_data.deinit(testing.allocator);

    std.debug.print("height: {}\n", .{jpeg_data.frame_info.height});
    std.debug.print("wid// Make sure the path is correct relative to src/zant.zigth: {}\n", .{jpeg_data.frame_info.width});
    for (jpeg_data.huffman_tables_dc) |table| {
        if (table != null) {
            std.debug.print("huffman_tables: {}\n", .{table.?.code_lengths.len});
        }
    }
    for (jpeg_data.huffman_tables_ac) |table| {
        if (table != null) {
            std.debug.print("huffman_tables: {}\n", .{table.?.code_lengths.len});
        }
    }

    // Verify some basic properties of the parsed JPEG
    try testing.expect(jpeg_data.frame_info.width > 0);
    try testing.expect(jpeg_data.frame_info.height > 0);
}
