//! This file contains the definition of the types of tensors that can be used in the library.
//! The tensorWrapper class is a wrapper around the tensor class so that it can be manipulated in the same way regardless of the true nature of the tensor itself.
//! - Tensor     -> used when weights ridden from the onnx file are not pre-processed in any way, regardless of the type of the weights.
//! - QuantTensor   -> used when weights ridden from the onnx file are quantized.
//! - ClustTensor   -> used when weights ridden from the onnx file are clustered.

const std = @import("std");
const zant = @import("../../zant.zig");

pub var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;

pub fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
}

pub const TensorType = enum {
    Tensor,
    QuantTensor,
    ClusterTensor,
    null,
};

// TODO: decide what to do with local functions (not public)

//------------------------------------------------------------------------------------------------------
/// TENSOR WRAPPER
///
/// TensorWrapper() is the superclass for all the possible implementation of a tensor (Tensor, QuantTensor, ClustTensor).
///
/// @param T:comptime type of the values in the tensor
pub fn TensorWrapper(comptime tensorType: anytype, comptime dataType: type) type {
    return struct {

        // tensor_type: TensorType,    // Type of the tensor

        // Interface fields
        //const tensorType = @TypeOf(tensor); // Type of the actual tensor implementation
        const tensorTypeInfo = @typeInfo(tensorType); // Type info of the actual tensor implementation
        ptr: *anyopaque, // Pointer to the actual tensor implementation

        // Interface functions
        initFn: *const fn (allocator: *const std.mem.Allocator) anyerror!tensorType,
        deinitFn: *const fn (ctx: *anyopaque) void,
        fromArrayFn: *const fn (allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) anyerror!tensorType,

        // TODO: verify if MagicalReturnType works, possible solution: make it public or use type instead of type
        // toArrayFn: *const fn (ctx: anyopaque, comptime dimension: usize) anyerror!MagicalReturnType(@This(), dataType, dimension),
        //toArrayFn: *const fn (ctx: anyopaque, comptime dimension: usize) anyerror!type, // TODO: anyopaque must be a pointer

        copyFn: *const fn (ctx: *anyopaque) anyerror!tensorType,
        fromShapeFn: *const fn (allocator: *const std.mem.Allocator, shape: []usize) anyerror!tensorType,
        fromConstBufferFn: *const fn (allocator: *const std.mem.Allocator, data: []const dataType, shape: []usize) tensorType,
        fillFn: *const fn (ctx: *anyopaque, inputArray: anytype, shape: []usize) anyerror!void,
        getSizeFn: *const fn (ctx: *anyopaque) usize,
        getFn: *const fn (ctx: *anyopaque, idx: usize) anyerror!dataType,
        setFn: *const fn (ctx: *anyopaque, idx: usize, value: dataType) anyerror!void,

        // GRANDE MURAGLIA CINESE

        getStridesFn: *const fn (ctx: *anyopaque) anyerror![]usize,
        infoFn: *const fn (ctx: *anyopaque) void,
        printFn: *const fn (ctx: *anyopaque) void,
        printMultidimFn: *const fn (ctx: *anyopaque) void,
        setToZeroFn: *const fn (ctx: *anyopaque) anyerror!void,
        slice_onnxFn: *const fn (ctx: *anyopaque, starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) anyerror!tensorType,
        ensure_4D_shapeFn: *const fn (shape: []const usize) callconv(.@"inline") anyerror![]usize,
        info_metalFn: *const fn (ctx: *anyopaque) void,

        /// Function used to create a new tensor wrapper for the provided tensor.
        pub fn createWrapper(ptr: anytype) !@This() {
            // const T = @TypeOf(ptr);
            // const ptr_info = @typeInfo(T);

            const gen = struct {
                pub fn init(allocator: *const std.mem.Allocator) !tensorType {
                    //const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.init(allocator);
                    //return tensorTypeInfo.Pointer.child.init(self, allocator);
                }

                pub fn deinit(ctx: *anyopaque) void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.deinit(self);
                }

                pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !tensorType {
                    return tensorType.fromArray(allocator, inputArray, shape);
                }

                // pub fn toArray(ctx: anyopaque, comptime dimension: usize) !MagicalReturnType(@This(), dataType, dimension) {
                //     // const self: tensorType = @ptrCast(@alignCast(ctx));
                //     const self: tensorType = @as(tensorType, ctx);
                //     return tensorType.toArray(self, dimension);
                // }

                pub fn copy(ctx: *anyopaque) !tensorType {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.copy(self);
                }

                pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) !tensorType {
                    return tensorType.fromShape(allocator, shape);
                }

                pub fn fromConstBuffer(allocator: *const std.mem.Allocator, data: []const dataType, shape: []const usize) tensorType {
                    return tensorType.fromConstBuffer(allocator, data, shape);
                }

                pub fn fill(ctx: *anyopaque, inputArray: anytype, shape: []usize) !void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.fill(self, inputArray, shape);
                }

                pub fn getSize(ctx: *anyopaque) usize {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.getSize(self);
                }

                pub fn get(ctx: *anyopaque, idx: usize) !dataType {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.get(self, idx);
                }

                pub fn set(ctx: *anyopaque, idx: usize, value: dataType) !void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.set(self, idx, value);
                }

                // GRANDE BARRIERA CORALLINA

                pub fn getStrides(ctx: *anyopaque) ![]usize {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.getStrides(self);
                }

                pub fn info(ctx: *anyopaque) void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.info(self);
                }

                pub fn print(ctx: *anyopaque) void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.print(self);
                }

                pub fn printMultidim(ctx: *anyopaque) void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.printMultidim(self);
                }

                pub fn setToZero(ctx: *anyopaque) !void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.setToZero(self);
                }

                pub fn slice_onnx(ctx: *anyopaque, starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !tensorType {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.slice_onnx(self, starts, ends, axes, steps);
                }

                pub inline fn ensure_4D_shape(shape: []const usize) ![]usize {
                    return tensorType.ensure_4D_shape(shape);
                }

                pub fn info_metal(ctx: *anyopaque) void {
                    const self: *tensorType = @ptrCast(@alignCast(ctx));
                    return tensorType.info_metal(self);
                }
            };

            return .{
                .ptr = ptr,
                .initFn = gen.init,
                .deinitFn = gen.deinit,
                .fromArrayFn = gen.fromArray,
                // .toArrayFn = gen.toArray,
                .copyFn = gen.copy,
                .fromShapeFn = gen.fromShape,
                .fromConstBufferFn = gen.fromConstBuffer,
                .fillFn = gen.fill,
                .getSizeFn = gen.getSize,
                .getFn = gen.get,
                .setFn = gen.set,

                // MURO DI TRUMP

                .getStridesFn = gen.getStrides,
                .infoFn = gen.info,
                .printFn = gen.print,
                .printMultidimFn = gen.printMultidim,
                .setToZeroFn = gen.setToZero,
                .slice_onnxFn = gen.slice_onnx,
                .ensure_4D_shapeFn = gen.ensure_4D_shape,
                .info_metalFn = gen.info_metal,
            };
        }

        // TODO: how should it be?
        // self: @This()    -   this should be fine
        // self: *@This()
        pub fn init(self: @This(), allocator: *const std.mem.Allocator) !tensorType {
            return self.initFn(allocator);
        }

        pub fn deinit(self: @This()) void {
            return self.deinitFn(self.ptr);
        }

        pub fn fromArray(self: @This(), allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {
            return self.fromArrayFn(allocator, inputArray, shape);
        }

        // pub fn toArray(self: @This(), comptime dimension: usize) !MagicalReturnType(self, dataType, dimension) {
        //     return self.toArrayFn(self.ptr, dimension);
        // }

        pub fn copy(self: @This()) !tensorType {
            return self.copyFn(self.ptr);
        }

        pub fn fromShape(self: @This(), allocator: *const std.mem.Allocator, shape: []usize) !tensorType {
            return self.fromShapeFn(allocator, shape);
        }

        pub fn fromConstBuffer(self: @This(), allocator: *const std.mem.Allocator, data: []const dataType, shape: []const usize) tensorType {
            return self.fromConstBufferFn(allocator, data, shape);
        }

        pub fn fill(self: @This(), inputArray: anytype, shape: []usize) !void {
            return self.fillFn(self.ptr, inputArray, shape);
        }

        pub fn getSize(self: @This()) usize {
            return self.getSizeFn(self.ptr);
        }

        pub fn get(self: @This(), idx: usize) !tensorType {
            return self.getFn(self.ptr, idx);
        }

        pub fn set(self: @This(), idx: usize, value: tensorType) !void {
            return self.setFn(self.ptr, idx, value);
        }

        // local function? magical return type
        fn MagicalReturnType(self: @This(), comptime DataType: type, comptime dim_count: usize) type {
            return self.MagicalReturnTypeFn(self.ptr, DataType, dim_count);
        }

        // MURO DI BERLINO
        // TODO: puntatori o meno a @This()?

        pub fn getStrides(self: @This()) ![]usize {
            return self.getStridesFn(self.ptr);
        }

        pub fn info(self: @This()) void {
            return self.infoFn(self.ptr);
        }

        pub fn print(self: @This()) void {
            return self.printFn(self.ptr);
        }

        pub fn printMultidim(self: @This()) void {
            return self.printMultidimFn(self.ptr);
        }

        pub fn setToZero(self: @This()) !void {
            return self.setToZeroFn(self.ptr);
        }

        pub fn slice_onnx(self: @This(), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !tensorType {
            return self.sliceONNXFn(self.ptr, starts, ends, axes, steps);
        }

        pub inline fn ensure_4D_shape(self: @This(), shape: []const usize) ![]usize {
            return self.ensure4DShapeFn(shape);
        }

        pub fn info_metal(self: @This()) void {
            return self.infoMetalFn(self.ptr);
        }

        // FUNZIONI ORIGINALI DI TENSOR

        // x   pub fn deinit(self: *Self) void
        // x   pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !Self
        //     pub fn toArray(self: Self, comptime dimension: usize) !MagicalReturnType(T, dimension) //
        //     //fn setAllocator(tensor: *Tensor(T), alloc: *const std.mem.Allocator) void
        // x   pub fn copy(self: *Self) !Tensor(T)
        // x   pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) !Self
        // x   pub fn fromConstBuffer(allocator: *const std.mem.Allocator, data: []const T, shape: []const usize) Self
        // x   pub fn fill(self: *Self, inputArray: anytype, shape: []usize) !void
        // x   pub fn getSize(self: *Self) usize
        // x   pub fn get(self: *const Self, idx: usize) !T
        // x   pub fn set(self: *Self, idx: usize, value: T) !void
        //     pub fn get_at(self: *const Self, indices: []const usize) !T
        //     pub fn set_at(self: *Self, indices: []const usize, value: T) !void
        //     fn constructMultidimensionalArray(allocator: *const std.mem.Allocator, comptime ElementType: type, data: []ElementType, shape: []usize, comptime depth: usize, comptime dimension: usize) !MagicalReturnType(ElementType, dimension - depth)
        //     fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type
        //     fn calculateProduct(slices: []usize) usize
        //     pub fn flatten_index(self: *const Self, indices: []const usize) !usize
        //     pub fn flatten_index_original(self: *const Self, indices: []const usize) !usize
        //     pub fn benchmark_flatten_index(self: *const Self, iterations: usize) struct { optimized: u64, original: u64 }
        //     pub fn slice(self: *Tensor(T), start_indices: []usize, slice_shape: []usize) !Tensor(T)
        //     fn copy_data_recursive(self: *Tensor(T), new_data: []T, new_data_index: *usize, start_indices: []usize, slice_shape: []usize, indices: []usize, dim: usize) !void
        //     fn get_flat_index(self: *Tensor(T), indices: []usize) !usize
        // x   pub fn getStrides(self: *Tensor(T)) ![]usize
        // x   pub fn info(self: *Self) void
        // x   pub fn print(self: *Self) void
        // x   pub fn printMultidim(self: *Self) void
        //     fn _printMultidimHelper(self: *Self, offset: usize, idx: usize) void
        // x   pub fn setToZero(self: *Self) !void
        // x   pub fn slice_onnx(self: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T)
        // x   pub inline fn ensure_4D_shape(shape: []const usize) ![]usize
        // x   pub fn info_metal(self: *Self) void

    };
}
