const std = @import("std");
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;
const allocator = @import("pkgAllocator").allocator;
const ReadyNode = @import("codeGen_predict.zig").ReadyNode;
const ModelOnnx = @import("onnx").ModelProto;

//Given an element from DataType Enum in onnx.zig returns the equivalent zig type
pub inline fn getType(data_type: DataType) !type {
    switch (data_type) {
        .FLOAT => {
            return f32;
        },
        .UINT8 => {
            return u8;
        },
        .INT8 => {
            return i8;
        },
        .UINT16 => {
            return u16;
        },
        .INT16 => {
            return i16;
        },
        .INT32 => {
            return i32;
        },
        .INT64 => {
            return i64;
        },
        .FLOAT16 => {
            return f16;
        },
        .DOUBLE => {
            return f64;
        },
        .UNIT32 => {
            return u32;
        },
        .UINT64 => {
            return u64;
        },
        else => return error.DataTypeNotAvailable,
    }
}
//Given an element from DataType Enum in onnx.zig returns the equivalent string of a zig type
pub inline fn getTypeString(data_type: DataType) ![]const u8 {
    switch (data_type) {
        .FLOAT => {
            return "f32";
        },
        .UINT8 => {
            return "u8";
        },
        .INT8 => {
            return "i8";
        },
        .UINT16 => {
            return "u16";
        },
        .INT16 => {
            return "i16";
        },
        .INT32 => {
            return "i32";
        },
        .INT64 => {
            return "i64";
        },
        .FLOAT16 => {
            return "f16";
        },
        .DOUBLE => {
            return "f64";
        },
        .UINT32 => {
            return "u32";
        },
        .UINT64 => {
            return "u64";
        },
        else => return error.DataTypeNotAvailable,
    }
}

//Returns the sanitized tensor's name, removes all non alphanumeric chars
pub inline fn getSanitizedName(name: []const u8) ![]const u8 {
    var sanitized = try allocator.alloc(u8, name.len);

    for (0..name.len) |i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(name[i]) or name[i] == '_') name[i] else '_';
    }

    std.debug.print("\nfrom {s} to {s} ", .{ name, sanitized });

    return sanitized;
}

//Returns a List of Ready nodes where all the input Tensor are set as ready
pub inline fn getComputableNodes(readyGraph: *std.ArrayList(ReadyNode)) !std.ArrayList(*ReadyNode) {
    var set: std.ArrayList(*ReadyNode) = std.ArrayList(*ReadyNode).init(allocator);

    for (readyGraph.items) |*node| {
        for (node.inputs.items) |input| {
            if (!input.ready) continue;
        }
        for (node.outputs.items) |output| {
            if (output.ready) return error.OutputReadyTooEarly;
        }
        try set.append(node);
    }

    return set;
}

//returns true if all the inputs and all the poutputs of a node are set as ready
pub inline fn isComputed(readyNode: *ReadyNode) !bool {
    for (readyNode.inputs.items) |input| {
        if (!input.ready) return false;
    }
    for (readyNode.outputs.items) |output| {
        if (!output.ready) return false;
    }
    return true;
}

//return true if the first parameter is an initializer
pub fn isInitializer(name: []const u8, initializers: []*TensorProto) bool {
    for (initializers) |init| {
        if (std.mem.eql(u8, init.name.?, name)) return true;
    }
    return false;
}

//return the relative TensorProto else error
pub fn getInitializer(name: []const u8, initializers: []*TensorProto) !*TensorProto {
    for (initializers) |init| {
        if (std.mem.eql(u8, init.name.?, name)) return init;
    }

    return error.NotExistingInitializer;
}
