const std = @import("std");
const readONNXFile = @import("../src/Utils/onnx_parser.zig").readONNXFile;

test "read ONNX file" {
    const allocator = std.testing.allocator;

    const file_path = "../src/tests/resources/feastconv_Opset16.onnx";

    // Attempt to read the file
    const onnx_data = try readONNXFile(allocator, file_path);
    defer allocator.free(onnx_data);

    // Check that the file was loaded correctly
    try std.testing.expect(onnx_data.len > 0);
}
