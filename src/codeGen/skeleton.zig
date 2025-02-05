const std = @import("std");
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;
const allocator = @import("pkgAllocator").allocator;

pub fn writeZigFile(file: std.fs.File, model: ModelOnnx) !void {
    const writer = file.writer();

    try writeLibraries(writer);

    try writeTensorsInit(writer, model);

    //try writePredict(writer, model);
}

inline fn writeLibraries(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const Tensor = @import("tensor").Tensor;
        \\ const TensMath = @import("tensor_m");
        \\ const pkgAllocator = @import("pkgAllocator");
        \\
        \\ const allocator = pkgAllocator.allocator;
    , .{});
}

inline fn writeTensorsInit(writer: std.fs.File.Writer, model: ModelOnnx) !void {

    //for each initializer of the onnx model we create the tensor
    for (model.graph.?.initializers) |tensorProtoInitializer| {
        // get the type and equivalent string
        const dataTypeString: []const u8 = try getTypeString(tensorProtoInitializer.data_type);
        //const dataType: type = try getType(tensorProtoInitializer.data_type);
        const name: []const u8 = try getSanitizedName(tensorProtoInitializer);

        try writer.print(
            \\
            \\
            \\ // ----------- initializing tensor_{s};
        , .{name});

        // ------ creating the shape
        try wrtiteTensorShape(writer, tensorProtoInitializer, name);

        // ------ creating the array
        try writeArray(writer, tensorProtoInitializer, name);

        //------ creating the tensor try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
        try writer.print(
            \\
            \\const tensor_{s} = Tensor({s}).fromArray(&allocator, & array_{s}, &shape_tensor_{s});
        , .{ name, dataTypeString, name, name });
    }
}

inline fn writeArray(writer: std.fs.File.Writer, t: *TensorProto, name: []const u8) !void {
    std.debug.print("\n  writeArray ", .{});

    const dataTypeString: []const u8 = try getTypeString(t.data_type);

    var size: i64 = 1;
    for (t.dims) |dims_i| {
        size = size * dims_i;
    }
    try writer.print(
        \\
        \\const array_{s} : [{d}]{s} = [_]{s}{{ 
    , .{ name, size, dataTypeString, dataTypeString });

    if (t.float_data) |d| {
        writeArrayData(writer, f32, d) catch return error.f32DataUnavailable;
    } else if (t.raw_data) |d| {
        writeArrayRawData(writer, t.data_type, d) catch return error.u8RawDataUnavailable;
    } else if (t.int32_data) |d| {
        writeArrayData(writer, i32, d) catch return error.i32DataUnavailable;
    } else if (t.int64_data) |d| {
        writeArrayData(writer, i64, d) catch return error.i64DataUnavailable;
    } else if (t.double_data) |d| {
        writeArrayData(writer, f64, d) catch return error.f64DataUnavailable;
    } else if (t.uint64_data) |d| {
        writeArrayData(writer, u64, d) catch return error.u64DataUnavailable;
    } else return error.DataTypeNotAvailable;

    try writer.print(
        \\}} ;
    , .{});
}

inline fn writeArrayRawData(writer: std.fs.File.Writer, data_type: DataType, data: []const u8) !void {
    std.debug.print("\n  from rawData to TypeData", .{});

    switch (data_type) {
        .FLOAT => {
            const float_slice = @as([*]const f32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            try writeArrayData(writer, f32, float_slice);
        },
        .UINT8 => {
            const uint_slice = @as([*]const u8, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 1)];
            try writeArrayData(writer, u8, uint_slice);
        },
        .INT8 => {
            const int_slice = @as([*]const i8, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 1)];
            try writeArrayData(writer, i8, int_slice);
        },
        .UINT16 => {
            const uint_slice = @as([*]const u16, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 2)];
            try writeArrayData(writer, u16, uint_slice);
        },
        .INT16 => {
            const int_slice = @as([*]const i16, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 2)];
            try writeArrayData(writer, i16, int_slice);
        },
        .INT32 => {
            const int_slice = @as([*]const i32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            try writeArrayData(writer, i32, int_slice);
        },
        .INT64 => {
            const int_slice = @as([*]const i64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            try writeArrayData(writer, i64, int_slice);
        },
        .FLOAT16 => {
            const float_slice = @as([*]const f16, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 2)];
            try writeArrayData(writer, f16, float_slice);
        },
        .DOUBLE => {
            const double_slice = @as([*]const f64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            try writeArrayData(writer, f64, double_slice);
        },
        .UINT32 => {
            const uint_slice = @as([*]const u32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            try writeArrayData(writer, u32, uint_slice);
        },
        .UINT64 => {
            const uint_slice = @as([*]const u64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            try writeArrayData(writer, u64, uint_slice);
        },
        else => {
            std.debug.print("\n data type {s} not supported for raw data access \n", .{@tagName(data_type)});
        },
    }
}

inline fn writeArrayData(writer: std.fs.File.Writer, comptime T: type, data: []const T) !void {
    std.debug.print("\n  writeArrayData ", .{});

    try writer.print(
        \\{}
    , .{data[0]});
    for (1..data.len) |i| {
        try writer.print(
            \\, {}
        , .{data[i]});
    }
}

inline fn wrtiteTensorShape(writer: std.fs.File.Writer, t: *TensorProto, name: []const u8) !void {
    try writer.print(
        \\
        \\
        \\const shape_tensor_{s} : [{}]usize = [_]usize{{ 
    , .{ name, t.dims.len });

    try writer.print(
        \\{}
    , .{t.dims[0]});
    for (1..t.dims.len) |i| {
        try writer.print(
            \\, {}
        , .{t.dims[i]});
    }

    try writer.print(
        \\}} ;
    , .{});
}

//Given an element from DataType Enum in onnx.zig returns the equivalent zig type
inline fn getType(data_type: DataType) !type {
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
inline fn getTypeString(data_type: DataType) ![]const u8 {
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
inline fn getSanitizedName(t: *TensorProto) ![]const u8 {
    const name = if (t.name) |n| n else return error.UnknownTensorName;

    var sanitized = try allocator.alloc(u8, name.len);

    for (0..name.len) |i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(name[i]) or name[i] == '_') name[i] else '_';
    }

    std.debug.print("\nfrom {s} to {s} ", .{ name, sanitized });

    return sanitized;
}

inline fn writePredict(writer: std.fs.File.Writer, model: ModelOnnx) !void {
    _ = try writer.print(
        \\
        \\pub fn predict() !Tensor {{
    , .{});

    _ = model;
    _ = try writer.print(
        \\
        \\}}
    , .{});
}
