const std = @import("std");
const ModelOnnx = @import("onnx").ModelProto;
const DataType = @import("onnx").DataType;
const TensorProto = @import("onnx").TensorProto;

pub fn writeZigFile(file: std.fs.File, model: ModelOnnx) !void {
    const writer = file.writer();

    try writeLibraries(writer);

    try writeTensorsInit(writer, model);

    //try writePredict(writer, model);
}

const tensor = @import("tensor");

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
        const name: []const u8 = try if (tensorProtoInitializer.name) |n| n else error.UnknownTensorName;

        try writer.print(
            \\
            \\
            \\ // ----------- initializing tensor_{s};
        , .{name});

        // ------ creating the shape
        try wrtiteTensorShape(writer, tensorProtoInitializer, name);

        // ------ creating the array
        try writeArray(writer, tensorProtoInitializer, name, dataTypeString);

        //------ creating the tensor try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
        try writer.print(
            \\
            \\const tensor_{s}({s}) = Tensor({s}).fromArray(&allocator, & array_{s}, &shape_tensor_{s});
        , .{ name, dataTypeString, dataTypeString, name, name });
    }
}

inline fn writeArray(writer: std.fs.File.Writer, t: *TensorProto, name: []const u8, dataTypeString: []const u8) !void {
    var size: i64 = 1;
    for (t.dims) |dims_i| {
        size = size * dims_i;
    }
    try writer.print(
        \\
        \\const array_{s} : [{d}]{s} = [_]usize{{ 
    , .{ name, size, dataTypeString });

    try switch (t.data_type) {
        .FLOAT => if (t.float_data) |d| writeArrayData(writer, f32, d) else return error.DataUnavailable,
        .UINT8 => if (t.raw_data) |d| writeArrayData(writer, u8, d) else return error.DataUnavailable,
        .INT32 => if (t.int32_data) |d| writeArrayData(writer, i32, d) else return error.DataUnavailable,
        .INT64 => if (t.int64_data) |d| writeArrayData(writer, i64, d) else return error.DataUnavailable,
        .DOUBLE => if (t.double_data) |d| writeArrayData(writer, f64, d) else return error.DataUnavailable,
        .UINT64 => if (t.uint64_data) |d| writeArrayData(writer, u64, d) else return error.DataUnavailable,
        else => return error.DataTypeNotAvailable,
    };

    try writer.print(
        \\}} ;
    , .{});
}

inline fn writeArrayData(writer: std.fs.File.Writer, comptime T: type, data: []const T) !void {
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
