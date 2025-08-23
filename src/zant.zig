// Standard library options for embedded targets
pub const std_options = struct {
    pub const page_size_min = 4096;
    pub const page_size_max = 4096;
    pub const log_level = .warn;
    pub const enable_segfault_handler = false;
};

// Conditional imports based on target
const builtin = @import("builtin");

pub const ImageToTensor = if (builtin.os.tag == .freestanding)
    struct {} // Empty struct for freestanding
else
    @import("ImageToTensor/imageToTensor.zig");

pub const ImageToTensorJpeg = if (builtin.os.tag == .freestanding)
    struct {} // Empty struct for freestanding
else
    @import("ImageToTensor/jpeg/jpegDecoder.zig");

pub const core = @import("Core/core.zig");
pub const utils = @import("Utils/utils.zig");

pub const onnx = if (builtin.os.tag == .freestanding)
    struct {} // Empty struct for freestanding
else
    @import("onnx/onnx.zig");

pub const weights_io = @import("zant/weights_io.zig");
