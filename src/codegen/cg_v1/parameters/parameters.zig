const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

// --- zant IR
const IR_utils = IR_zant.utils;
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;

const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_zant.tensorZant_lib.tensorMap;

/// Writes the Zig code required to initialize all tensor initializers in the ONNX model.
/// This function generates declarations and definitions for each tensor.
///
/// - `writer`: The file writer to output generated code.
pub inline fn write_parameters(writer: std.fs.File.Writer) !void {

    //importing the libraries
    try write_libraries_parameters(writer);

    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         Initializing Weights and Biases         +
        \\ // ---------------------------------------------------
    , .{});

    try write_initilizers(writer);

    try writer.print(
        \\
        \\
        \\ // -----------------------------------------
        \\ // +         Initializing constants        +
        \\ // -----------------------------------------
    , .{});

    try write_constantTensors(writer);
}

fn write_initilizers(writer: std.fs.File.Writer) !void {
    const initializers: []TensorZant = try IR_utils.getInitializers(tensorZantMap);

    // Iterate over all initializers in the ONNX model and generate code
    for (initializers) |*initializer| {
        const name: []const u8 = try initializer.getNameSanitized();

        try writer.print(
            \\
            \\
            \\ // ----------- Initializing tensor_{s};
        , .{name});

        // Generate the shape array for the tensor
        try wrtiteTensorShape(writer, initializer);

        // Generate the data array for the tensor
        try writeArray(writer, initializer);

        // Create the tensor instance
        try writer.print(
            \\
            \\pub const tensor_{s} = Tensor({s}).fromConstBuffer(&allocator, &array_{s}, &shape_tensor_{s});
        , .{ name, initializer.ty.toString(), name, name });
    }
}

fn write_constantTensors(writer: std.fs.File.Writer) !void {
    const constants: []TensorZant = try IR_utils.getConstants(tensorZantMap);

    if (constants.len == 0) {
        try writer.print(
            \\
            \\
            \\ // no Constant Tensors are present;
        , .{});

        return;
    }
    // Iterate over all initializers in the ONNX model and generate code
    for (constants) |*constant_tensors| {
        const name: []const u8 = try constant_tensors.getNameSanitized();

        try writer.print(
            \\
            \\
            \\ // ----------- Initializing Constant tensor_{s};
        , .{name});

        // Generate the shape array for the tensor
        try wrtiteTensorShape(writer, constant_tensors);

        // Generate the data array for the tensor
        try writeArray(writer, constant_tensors);

        // Create the tensor instance
        try writer.print(
            \\
            \\pub const tensor_{s} = Tensor({s}).fromConstBuffer(&allocator, &array_{s}, &shape_tensor_{s});
        , .{ name, constant_tensors.ty.toString(), name, name });
    }
}

/// Writes the required library imports to the generated Zig file for input tensor.
///
/// This function ensures that the necessary standard and package libraries are
/// imported into the generated Zig source file.
///
/// # Parameters
/// - `writer`: A file writer used to write the import statements.
///
/// # Errors
/// This function may return an error if writing to the file fails.
fn write_libraries_parameters(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const zant = @import("zant");
        \\ const Tensor = zant.core.tensor.Tensor;
        \\ const pkgAllocator = zant.utils.allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\
    , .{});
}

/// Writes the shape array for a tensor initializer.
///
/// - `writer`: The file writer to output generated code.
/// - `t`: The tensor initializer.
/// - `name`: The sanitized name of the tensor.
pub inline fn wrtiteTensorShape(writer: std.fs.File.Writer, tz: *TensorZant) !void {
    try writer.print(
        \\
        \\
        \\const shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{ try tz.getNameSanitized(), tz.getShape().len });

    for (tz.getShape(), 0..) |dim_i, i| {
        if (i > 0) try writer.print(
            \\,
        , .{});

        try writer.print(
            \\ {}
        , .{dim_i});
    }

    try writer.print(
        \\}} ;
    , .{});
}

/// Writes the array for a tensor initializer based on its data type.
///
/// - `writer`: The file writer to output generated code.
/// - `tz`: The tensor initializer.
pub inline fn writeArray(writer: std.fs.File.Writer, tz: *TensorZant) !void {
    // std.log.info("\n[writeArray] Processing tensor: {s}, DataType: {any}", .{ name, t.data_type });

    try writer.print(
        \\
        \\const array_{s} : [{d}]{s} linksection(".rodata") = [_]{s}{{
    , .{ try tz.getNameSanitized(), tz.getSize(), tz.ty.toString(), tz.ty.toString() });

    switch (tz.ty) {
        .f16 => writeArrayData(writer, f16, tz.ptr.?.get_data_as(f16)) catch return error.f14DataUnavailable,
        .f32 => writeArrayData(writer, f32, tz.ptr.?.get_data_as(f32)) catch return error.f32DataUnavailable,
        .f64 => writeArrayData(writer, f64, tz.ptr.?.get_data_as(f64)) catch return error.f64DataUnavailable,
        .i8 => writeArrayData(writer, i8, tz.ptr.?.get_data_as(i8)) catch return error.i8DataUnavailable,
        .i16 => writeArrayData(writer, i16, tz.ptr.?.get_data_as(i16)) catch return error.i16DataUnavailable,
        .i32 => writeArrayData(writer, i32, tz.ptr.?.get_data_as(i32)) catch return error.i32DataUnavailable,
        .i64 => writeArrayData(writer, i64, tz.ptr.?.get_data_as(i64)) catch return error.i64DataUnavailable,
        .u8 => writeArrayData(writer, u8, tz.ptr.?.get_data_as(u8)) catch return error.u8DataUnavailable,
        .u16 => writeArrayData(writer, u16, tz.ptr.?.get_data_as(u16)) catch return error.u16DataUnavailable,
        .u32 => writeArrayData(writer, u32, tz.ptr.?.get_data_as(u32)) catch return error.u32DataUnavailable,
        .u64 => writeArrayData(writer, u64, tz.ptr.?.get_data_as(u64)) catch return error.u64DataUnavailable,
        .bool => writeArrayData(writer, bool, tz.ptr.?.get_data_as(bool)) catch return error.boolDataUnavailable,
        .undefined => try writer.print(
            \\ undefined
        , .{}),
    }

    try writer.print(
        \\}} ;
    , .{});
}

/// Writes an array of tensor data.
///
/// - `writer`: The file writer to output generated code.
/// - `T`: The type of data in the tensor.
/// - `data`: The data array.
pub inline fn writeArrayData(writer: std.fs.File.Writer, comptime T: type, data: []const T) !void {
    for (0..data.len) |i| {
        if (i > 0) try writer.print(
            \\,
        , .{});
        try writer.print(
            \\ {}
        , .{data[i]});
    }
}
