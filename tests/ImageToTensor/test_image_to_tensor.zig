const std = @import("std");
const zant = @import("zant");
const ImageToTensor = zant.ImageToTensor;
const pgkAllocator = zant.utils.allocator;
const imageToRGB = ImageToTensor.imageToRGB;
const imageToYCbCr = ImageToTensor.imageToYCbCr;
const imageToGray = ImageToTensor.imageToGray;

const norm_type: usize = 1;

fn getBound(norm: usize) f32 {
    if (norm == 0) {
        return 0.0;
    } else {
        return -1.0;
    }
}

const bound = getBound(norm_type);

test "jpeg baseline standard" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/baseline.jpg";
    var image = try imageToRGB(&allocator, image_path, norm_type, f32);
    defer image.deinit();

    var imagYcbCr = try imageToYCbCr(&allocator, image_path, norm_type, f32);
    defer imagYcbCr.deinit();

    var imageY = try imageToGray(&allocator, image_path, norm_type, f32);
    defer imageY.deinit();

    // Check that all values are between 0 and 1

    for (image.data) |value| {
        try std.testing.expect(value >= bound and value <= 1);
    }
}

test "jpeg restart interval" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/restart_intervals.jpg";
    var image = try imageToRGB(&allocator, image_path, norm_type, f32);
    defer image.deinit();

    var imagYcbCr = try imageToYCbCr(&allocator, image_path, norm_type, f32);
    defer imagYcbCr.deinit();

    var imageY = try imageToGray(&allocator, image_path, norm_type, f32);
    defer imageY.deinit();

    // Check that all values are between 0 and 1

    for (image.data) |value| {
        try std.testing.expect(value >= bound and value <= 1);
    }
}

test "jpeg subsampling" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/subsampling.jpg";
    var image = try imageToRGB(&allocator, image_path, norm_type, f32);
    defer image.deinit();

    var imagYcbCr = try imageToYCbCr(&allocator, image_path, norm_type, f32);
    defer imagYcbCr.deinit();

    var imageY = try imageToGray(&allocator, image_path, norm_type, f32);
    defer imageY.deinit();

    // Check that all values are between 0 and 1

    for (image.data) |value| {
        try std.testing.expect(value >= bound and value <= 1);
    }
}

test "jpeg t1" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t1.jpg";
    var image = try imageToRGB(&allocator, image_path, norm_type, f32);
    defer image.deinit();

    var imagYcbCr = try imageToYCbCr(&allocator, image_path, norm_type, f32);
    defer imagYcbCr.deinit();

    var imageY = try imageToGray(&allocator, image_path, norm_type, f32);
    defer imageY.deinit();

    // Check that all values are between 0 and 1

    for (image.data) |value| {
        try std.testing.expect(value >= bound and value <= 1);
    }
}

test "jpeg t2" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t2.jpg";
    var image = try imageToRGB(&allocator, image_path, norm_type, f32);
    defer image.deinit();

    var imagYcbCr = try imageToYCbCr(&allocator, image_path, norm_type, f32);
    defer imagYcbCr.deinit();

    var imageY = try imageToGray(&allocator, image_path, norm_type, f32);
    defer imageY.deinit();

    // Check that all values are between 0 and 1

    for (image.data) |value| {
        try std.testing.expect(value >= bound and value <= 1);
    }
}

test "jpeg t3" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t3.jpg";
    var image = try imageToRGB(&allocator, image_path, norm_type, f32);
    defer image.deinit();

    var imagYcbCr = try imageToYCbCr(&allocator, image_path, norm_type, f32);
    defer imagYcbCr.deinit();

    var imageY = try imageToGray(&allocator, image_path, norm_type, f32);
    defer imageY.deinit();

    // Check that all values are between 0 and 1

    for (image.data) |value| {
        try std.testing.expect(value >= bound and value <= 1);
    }
}

test "jpeg t4" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/t4.jpg";
    var image = try imageToRGB(&allocator, image_path, norm_type, f32);
    defer image.deinit();

    var imagYcbCr = try imageToYCbCr(&allocator, image_path, norm_type, f32);
    defer imagYcbCr.deinit();

    var imageY = try imageToGray(&allocator, image_path, norm_type, f32);
    defer imageY.deinit();

    // Check that all values are between 0 and 1
    for (image.data) |value| {
        try std.testing.expect(value >= bound and value <= 1);
    }
}
