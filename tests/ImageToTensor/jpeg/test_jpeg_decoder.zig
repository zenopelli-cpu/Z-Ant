const std = @import("std");
const zant = @import("zant");
const jpeg = zant.ImageToTensor.jpeg;
const pgkAllocator = zant.utils.allocator;
const debug_jpegToRGB = jpeg.debug_jpegToRGB;
const debug_jpegToYCbCr = jpeg.debug_jpegToYCbCr;
const debug_jpegToGrayscale = jpeg.debug_jpegToGrayscale;

test "jpeg baseline standard" {
    std.debug.print("test jpeg baseline standard\n", .{});
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/baseline.jpg";
    try debug_jpegToRGB(&allocator, image_path);

    try debug_jpegToYCbCr(&allocator, image_path);

    try debug_jpegToGrayscale(&allocator, image_path);
}

test "jpeg restart interval" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/restart_intervals.jpg";
    try debug_jpegToRGB(&allocator, image_path);

    try debug_jpegToYCbCr(&allocator, image_path);

    try debug_jpegToGrayscale(&allocator, image_path);
}

test "jpeg subsampling" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/subsampling.jpg";
    try debug_jpegToRGB(&allocator, image_path);

    try debug_jpegToYCbCr(&allocator, image_path);

    try debug_jpegToGrayscale(&allocator, image_path);
}

test "jpeg t1" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t1.jpg";
    try debug_jpegToRGB(&allocator, image_path);

    try debug_jpegToYCbCr(&allocator, image_path);
    try debug_jpegToGrayscale(&allocator, image_path);
}

test "jpeg t2" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t2.jpg";
    try debug_jpegToRGB(&allocator, image_path);

    try debug_jpegToYCbCr(&allocator, image_path);

    try debug_jpegToGrayscale(&allocator, image_path);
}

test "jpeg t3" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t3.jpg";
    try debug_jpegToRGB(&allocator, image_path);

    try debug_jpegToYCbCr(&allocator, image_path);

    try debug_jpegToGrayscale(&allocator, image_path);
}

test "jpeg t4" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t4.jpg";
    try debug_jpegToRGB(&allocator, image_path);

    try debug_jpegToYCbCr(&allocator, image_path);

    try debug_jpegToGrayscale(&allocator, image_path);
}
