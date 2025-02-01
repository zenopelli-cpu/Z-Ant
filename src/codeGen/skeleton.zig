const std = @import("std");
//const ModelOnnx = @import("onnx").ModelProto;

pub fn writeZigFile(file: std.fs.File) !void { //, model: ModelOnnx
    const writer = file.writer();

    _ = try writer.print(
        \\const std = @import("std");
        \\{s}
        \\
        \\ //Initializing wheight tensors
        \\{s}
        \\
        \\ //predictive function
        \\pub fn predict() !void {{
        \\  {s}
        \\}}
    , .{ try writeLibraries(), try writeLibraries(), try writeLibraries() });
}

const tensor = @import("tensor");

inline fn writeLibraries() ![]const u8 {
    const libraries =
        \\
        \\ const tensor = @import("tensor");
        \\ const TensMath = @import("tensor_m");
    ;

    return libraries;
}
