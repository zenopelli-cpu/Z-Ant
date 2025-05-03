const std = @import("std");
const zant = @import("zant");
const ImageToTensor = zant.ImageToTensor;
const pgkAllocator = zant.utils.allocator;
const imageToRGB = ImageToTensor.imageToRGB;

test "jpeg baseline standard" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/baseline.jpg";
    var image = try ImageToTensor.imageToRGB(&allocator, image_path, 0, f32);
    defer image.deinit();

    // Check that all values are between 0 and 1
    for (image.data) |value| {
        try std.testing.expect(value >= 0 and value <= 1);
    }
}

test "jpeg subsampling" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/subsampling.jpg";
    var image = try ImageToTensor.imageToRGB(&allocator, image_path, 0, f32);
    defer image.deinit();

    // Check that all values are between 0 and 1
    for (image.data) |value| {
        try std.testing.expect(value >= 0 and value <= 1);
    }
}

test "jpeg restart interval" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t3.jpg";
    var image = try ImageToTensor.imageToRGB(&allocator, image_path, 0, f32);
    defer image.deinit();

    // Check that all values are between 0 and 1
    for (image.data) |value| {
        try std.testing.expect(value >= 0 and value <= 1);
    }
}
