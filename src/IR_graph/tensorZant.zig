const std = @import("std");
const zant = @import("../zant.zig");
const Tensor = zant.core.tensor.Tensor;
pub const AnyTensor = zant.core.tensor.AnyTensor;

const utils = @import("utils.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const ValueInfoProto = onnx.ValueInfoProto;

const allocator = std.heap.page_allocator;

pub const TensorType = enum {
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

pub const TensorCategory = enum {
    input,
    output,
    initializer,
    link,
};

pub const TensorZant = struct {
    name: []const u8,
    ty: TensorType,
    tc: TensorCategory,
    ptr: ?*AnyTensor,
    shape: []usize,

    pub fn init(name: []const u8, tensorProto: ?*TensorProto, value_info: ?*ValueInfoProto, tensorCategory: TensorCategory) !TensorZant {
        std.debug.print("\n ----------- init({s}, {s}, model) ", .{ name, if (tensorProto) |_| "tp" else "null" });

        var tensor: ?AnyTensor = null;
        var shape_i64: []i64 = undefined;
        var shape_usize: []usize = undefined;

        if (tensorProto) |tp| { //if the tensorProto is given it means that the tensor we are initializing is
            tensor = try utils.protoTensor2AnyTensor(tp); //create the ptr to AnyTensor
            shape_usize = tensor.?.get_shape(); //saves the shape
        } else if (value_info) |vi| {
            shape_i64 = if (utils.getTensorShapeFromValueInfo(vi)) |s| s else { //search for the shape
                std.debug.print("\n ERROR: {s} value_info shape not found ", .{name});
                return error.shapeNotfound;
            };
            shape_usize = try utils.i64SliceToUsizeSlice(shape_i64); //saves the shape
        } else {
            std.debug.print("\n ERROR: {s} not found ", .{name});
            return error.shapeNotfound;
        }

        std.debug.print("\n                shape:{any} ", .{shape_usize});

        return TensorZant{
            .name = name,
            .ty = if (tensor) |t| utils.getAnyTensorType(t) else TensorType.undefined, //if .ty is set to undefined it means that it is a "link" tensor between 2 nodes, the .ty must be set when the nodes are created
            .tc = tensorCategory,
            .ptr = if (tensor) |t| @constCast(&t) else null,
            .shape = shape_usize,
        };
    }

    pub fn get_shape(self: *TensorZant) []usize {
        return self.ptr.get_shape();
    }

    pub fn get_stride(self: *TensorZant) ![]usize {
        return try self.ptr.get_stride();
    }

    pub fn set_tensorType(self: *TensorZant, ty: TensorType) void {
        if (ty == TensorType.undefined) {
            std.debug.print("\n ERROR: illegal behavior! you cannot set a tensor type to undefined! ", .{});
            return error.illegalBehavior;
        }
        self.ty = ty;
    }
};

// ----------------------- HASH MAP -----------------------
pub var tensorMap: std.StringHashMap(TensorZant) = std.StringHashMap(TensorZant).init(allocator);

// Populates tensorHashMap with the tensors used in the onnx graph, where the key is the name of the tensor

pub fn initialize_tensorZantMap(modelProto: *ModelProto) !void {
    std.debug.print("\n ------ initialize_tensorZantMap ----- ", .{});

    const protoGraph = try if (modelProto.graph) |graph| graph else error.GraphNotAvailable;

    //adding initializers to the hash map
    std.debug.print("\n ---------- initializers ", .{});

    for (protoGraph.initializers) |init_ptr| { //initializer : *TensorProto,
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            init_ptr.name.?,
            init_ptr,
            null,
            TensorCategory.initializer,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding inputs to the hash map
    std.debug.print("\n ---------- inputs ", .{});

    for (protoGraph.inputs) |inputs_ptr| { //inputs : *ValueInfoProto,
        if (tensorMap.getPtr(inputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            inputs_ptr.name.?,
            null,
            inputs_ptr,
            TensorCategory.input,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding outputs to the hash map
    std.debug.print("\n ---------- outputs ", .{});

    for (protoGraph.outputs) |outputs_ptr| { //outputs : *ValueInfoProto,
        if (tensorMap.getPtr(outputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            outputs_ptr.name.?,
            null,
            outputs_ptr,
            TensorCategory.output,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    std.debug.print("\n ---------- nodes ", .{});
    //adding all the nodes inputs and outputs
    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto
        for (node.input) |input_name| {
            //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
            if (tensorMap.getPtr(input_name) != null) continue;
            //create the readyTensor
            const tensorZant: TensorZant = try TensorZant.init(input_name, null, utils.getValueInfoTensorFromGraphInfo(input_name, protoGraph), TensorCategory.link);
            //add the readyTensor to the HashMap
            try tensorMap.put(tensorZant.name, tensorZant);
        }
        for (node.output) |output_name| {
            //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
            if (tensorMap.getPtr(output_name) != null) continue;
            //create the readyTensor
            const tensorZant: TensorZant = try TensorZant.init(output_name, null, utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph), TensorCategory.link);
            //add the readyTensor to the HashMap
            try tensorMap.put(tensorZant.name, tensorZant);
        }
    }
}
