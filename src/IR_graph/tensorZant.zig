const std = @import("std");
const zant = @import("../zant.zig");
const Tensor = zant.core.tensor.Tensor;
pub const AnyTensor = zant.core.tensor.AnyTensor;

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

    pub fn protoTensor2Tensor(proto: TensorProto) !AnyTensor {
        // Allocate shape array
        var shape = try allocator.alloc(usize, proto.dims.len);
        for (proto.dims, 0..) |dim, i| {
            if (dim < 0) {
                return error.NegativeDimension;
            }
            shape[i] = @intCast(dim);
        }
        defer allocator.free(shape);

        if (proto.float_data) |float_data| {
            const tensor = try Tensor(f32).fromArray(&allocator, float_data, shape);
            return AnyTensor{ .f32 = @constCast(&tensor) };
        } else if (proto.int32_data) |int32_data| {
            const tensor = try Tensor(i32).fromArray(&allocator, int32_data, shape);
            return AnyTensor{ .i32 = @constCast(&tensor) };
        } else if (proto.int64_data) |int64_data| {
            const tensor = try Tensor(i64).fromArray(&allocator, int64_data, shape);
            return AnyTensor{ .i64 = @constCast(&tensor) };
        } else if (proto.double_data) |double_data| {
            const tensor = try Tensor(f64).fromArray(&allocator, double_data, shape);
            return AnyTensor{ .f64 = @constCast(&tensor) };
        } else if (proto.uint64_data) |uint64_data| {
            const tensor = try Tensor(u64).fromArray(&allocator, uint64_data, shape);
            return AnyTensor{ .u64 = @constCast(&tensor) };
        } else if (proto.uint16_data) |uint16_data| {
            const tensor = try Tensor(u16).fromArray(&allocator, uint16_data, shape);
            return AnyTensor{ .u16 = @constCast(&tensor) };
        } else {
            return error.UnsupportedDataType;
        }
    }
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
