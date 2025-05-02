const std = @import("std");
const zant = @import("../zant.zig");
const Tensor = zant.core.tensor.Tensor;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

const allocator = std.heap.page_allocator;

const TensorType = enum {
    // Floating point types
    f16,
    f32,
    f64,

    // Signed integer types
    i8,
    i16,
    i32,
    i64,

    // Unsigned integer types
    u8,
    u16,
    u32,
    u64,

    // Boolean (often used for masks)
    bool,

    undefined,
};

pub const TensorZant = struct {
    name: []const u8,
    ty: TensorType,
    ptr: ?*anyopaque,

    pub fn init(name: []const u8, tensorProto: ?*TensorProto, nodeProto: ?*NodeProto, graphProto: ?*GraphProto) !TensorZant {
        _ = tensorProto;
        _ = nodeProto;
        _ = graphProto;

        // const tensor_ptr = if (tensorProto) |tp| protoTensor2Tensor(tp) else null;

        //create a Tensor(T)
        return TensorZant{
            .name = name,
            .ty = TensorType.undefined,
            .ptr = null,
        };
    }

    pub fn get_shape(self: TensorZant) void { //TODO: return []usize
        _ = self;
    }

    pub fn get_stride(self: TensorZant) void { //TODO: return []usize
        _ = self;
    }

    // pub fn protoTensor2Tensor(T: type, proto: TensorProto) !Tensor(T) {
    //     // Type Check
    //     if (!isMatchingType(T, proto.data_type)) {
    //         return error.InvalidDataType;
    //     }

    //     // Allocate shape array
    //     var shape = try allocator.alloc(usize, proto.dims.len);
    //     for (proto.dims, 0..) |dim, i| {
    //         if (dim < 0) {
    //             return error.NegativeDimension;
    //         }
    //         shape[i] = @intCast(dim);
    //     }

    //     // Compute total size
    //     var size: usize = 1;
    //     for (shape) |dim| {
    //         size *= dim;
    //     }

    //     // Allocate data array
    //     var data = try allocator.alloc(T, size);
    //     // Fill data
    //     if (proto.raw_data) |raw| {
    //         // Fill from raw_data
    //         const needed_bytes = size * @sizeOf(T);
    //         if (raw.len != needed_bytes) {
    //             return error.RawDataSizeMismatch;
    //         }
    //     } else {
    //         // Fill from typed fields
    //         if (T == f32) {
    //             data = proto.float_data.?;
    //         }
    //         if (T == i32) {
    //             data = proto.int32_data.?;
    //         }
    //         if (T == i64) {
    //             data = proto.int64_data.?;
    //         }
    //         if (T == f64) {
    //             data = proto.double_data.?;
    //         }
    //     }

    //     // Return the Tensor
    //     return Tensor(T){
    //         .data = data,
    //         .size = size,
    //         .shape = shape,
    //         .allocator = &allocator,
    //     };
    // }

    // fn isMatchingType(comptime T: type, data_type: DataType) bool {
    //     return switch (data_type) {
    //         .FLOAT => T == f32,
    //         .INT32 => T == i32,
    //         .INT64 => T == i64,
    //         .DOUBLE => T == f64,
    //         .UINT64 => T == u64,
    //         .UINT16 => T == u16,
    //         else => false,
    //     };
    // }
};

// ----------------------- HASH MAP -----------------------
pub var tensorMap: std.StringHashMap(TensorZant) = std.StringHashMap(TensorZant).init(allocator);

// Populates tensorHashMap with the tensors used in the onnx graph, where the key is the name of the tensor

pub fn initialize_tensorZantMap(modelProto: *ModelProto) !void {
    const protoGraph = try if (modelProto.graph) |graph| graph else error.GraphNotAvailable;

    //adding initializers to the hash map
    for (protoGraph.initializers) |init_ptr| { //initializer : *TensorProto,
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(init_ptr.name.?, init_ptr, null, null);
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding all the nodes inputs and outputs
    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto
        for (node.input) |input_name| {
            //create the readyTensor
            const tensorZant: TensorZant = try TensorZant.init(input_name, null, null, null);
            //add the readyTensor to the HashMap
            try tensorMap.put(tensorZant.name, tensorZant);
        }
        for (node.output) |output_name| {
            //create the readyTensor
            const tensorZant: TensorZant = try TensorZant.init(output_name, null, null, null);
            //add the readyTensor to the HashMap
            try tensorMap.put(tensorZant.name, tensorZant);
        }
    }
}
