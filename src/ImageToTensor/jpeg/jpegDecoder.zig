const std = @import("std");
const alg = @import("jpegAlgorithms.zig");
const parser = @import("jpegParser.zig");
const bmp = @import("../writerBMP.zig");
const utils = @import("../utils.zig");

pub const SegmentReader = utils.SegmentReader;
const ColorChannels = utils.ColorChannels;
const JpegData = parser.JpegData;
const MCU = alg.MCU;

pub fn jpegToYCbCr(segment_reader: *SegmentReader, allocator: *const std.mem.Allocator) !ColorChannels {

    // parse the Jpeg file
    var header = try parser.jpegParser(allocator, segment_reader);
    defer header.deinit(allocator);

    // Decode Huffman entropy data
    const mcus = try allocator.alloc(MCU, header.mcu_true_height * header.mcu_true_width);
    defer {
        for (mcus) |mcu| {
            allocator.free(mcu.y);
            allocator.free(mcu.cb);
            allocator.free(mcu.cr);
        }
        allocator.free(mcus);
    }

    // Decode Huffman entropy data
    try alg.decodeHuffmanData(header, allocator, mcus);

    // Dequantize MCUS
    try alg.dequantize(header, mcus);

    // Inverse Discrete Cosine Transform for each MCU
    try alg.inverseDCT(header, mcus);

    for (0..mcus.len) |i| {
        for (0..64) |j| {
            for (0..header.frame_info.components_num) |k| {
                mcus[i].get(k).*[j] += 128;
            }
        }
    }
    // convert to 3 color channels:
    const channels = try alg.writeChannels(header, mcus, allocator);

    return channels;
}

pub fn jpegToRGB(segment_reader: *SegmentReader, allocator: *const std.mem.Allocator) !ColorChannels {
    // parse the Jpeg file
    var header = try parser.jpegParser(allocator, segment_reader);
    defer header.deinit(allocator);
    // Decode Huffman entropy data
    const mcus = try allocator.alloc(MCU, header.mcu_true_height * header.mcu_true_width);
    defer {
        for (mcus) |mcu| {
            allocator.free(mcu.y);
            allocator.free(mcu.cb);
            allocator.free(mcu.cr);
        }
        allocator.free(mcus);
    }

    // Decode Huffman entropy data
    try alg.decodeHuffmanData(header, allocator, mcus);

    // Dequantize MCUS
    try alg.dequantize(header, mcus);

    // Inverse Discrete Cosine Transform for each MCU
    try alg.inverseDCT(header, mcus);

    // ycbcr to rgb
    try alg.yCbCrToRgb(header, mcus);

    // convert to 3 color channels:
    return try alg.writeChannels(header, mcus, allocator);
}

pub fn jpegToGray(segment_reader: *SegmentReader, allocator: *const std.mem.Allocator) !ColorChannels {
    var yCbCrChannels: ColorChannels = try jpegToYCbCr(segment_reader, allocator);
    // convert to 3 cde the image using the appropriate decoderolor channels:
    @memcpy(&yCbCrChannels.ch2, yCbCrChannels.ch1);
    @memcpy(&yCbCrChannels.ch3, yCbCrChannels.ch1);
    return yCbCrChannels;
}
