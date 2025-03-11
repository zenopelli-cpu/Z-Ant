//! The aim of this file is that, given a onnx oprator (https://onnx.ai/onnx/operators/),
//! it returns the onnx.ModelProto containing only one node for the operation

const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const onnx = zant.onnx;
const codeGen = zant.codeGen;

//------------operation structs
//Add
pub const AddStruct = struct {
    name: []const u8,
    inputShape: []i64,
    //other...
};

//Concat
pub const ConcatStruct = struct {
    name: []const u8,
    inputShape: []i64,
    //other...
};

//Conv
pub const ConvStruct = struct {
    name: []const u8,
    inputShape: []i64,
    //other...
};

//------------union of all the operation struct
pub const OpStruct = union(enum) {
    Add: AddStruct,
    Concat: ConcatStruct,
    Conv: ConvStruct,
    Default: AddStruct, // delete this, just for DEBUG
};

pub fn oneOpModel(
    op_struct: OpStruct,
) !onnx.ModelProto {

    // ---------- GRAPH ATTRIBUTES INITIALIZATION ----------
    var graph_nodes = std.ArrayList(*onnx.NodeProto).init(allocator);
    defer graph_nodes.deinit();
    var graph_initializers = std.ArrayList(*onnx.TensorProto).init(allocator);
    defer graph_initializers.deinit();
    var graph_inputs = std.ArrayList(*onnx.ValueInfoProto).init(allocator);
    defer graph_inputs.deinit();
    var graph_outputs = std.ArrayList(*onnx.ValueInfoProto).init(allocator);
    defer graph_outputs.deinit();
    var graph_value_infos = std.ArrayList(*onnx.ValueInfoProto).init(allocator);
    defer graph_value_infos.deinit();

    // ---------- NODE ATTRIBUTES INITIALIZATION ----------
    var node_inputs = std.ArrayList([]const u8).init(allocator);
    defer node_inputs.deinit();
    var node_outputs = std.ArrayList([]const u8).init(allocator);
    defer node_outputs.deinit();
    var node_attributes = std.ArrayList(*onnx.AttributeProto).init(allocator);
    defer node_attributes.deinit();

    const node_ptr = try allocator.create(onnx.NodeProto);
    node_ptr.* = onnx.NodeProto{
        .name = null, // ?[]const u8,
        .op_type = undefined, //[]const u8,
        .domain = null, // ?[]const u8,
        .input = undefined, //[][]const u8,
        .output = undefined, // [][]const u8,
        .attribute = undefined, // []*AttributeProto,
    };

    switch (op_struct) {
        .Add => |details| {
            node_ptr.*.name = try std.mem.concat(allocator, u8, &[_][]const u8{
                "node_",
                details.name,
            });
            node_ptr.*.op_type = try std.mem.concat(allocator, u8, &[_][]const u8{
                "Add",
            });
            try node_inputs.append("input_Add");
            try node_outputs.append("output_Add");
        },
        .Concat => |details| {
            node_ptr.*.name = details.name;
            node_ptr.*.op_type = "Concat";
            try node_inputs.append("input_Concat");
            try node_outputs.append("output_Concat");
        },
        .Conv => |details| {
            node_ptr.*.name = details.name;
            node_ptr.*.op_type = "Conv";
            try node_inputs.append("input_Conv");
            try node_outputs.append("output_Conv");
        },
        else => return error.OpNotAvalilable,
    }

    //completing and append the only node of the graph
    node_ptr.*.input = try node_inputs.toOwnedSlice();
    node_ptr.*.output = try node_outputs.toOwnedSlice();
    node_ptr.*.attribute = try node_attributes.toOwnedSlice();
    try graph_nodes.append(node_ptr);

    //graph creation
    const graph_ptr = try allocator.create(onnx.GraphProto);
    graph_ptr.* = onnx.GraphProto{ //?*GraphProto,
        .name = null, //?[]const u8,
        .nodes = try graph_nodes.toOwnedSlice(), // []*NodeProto,
        .initializers = try graph_initializers.toOwnedSlice(), // []*TensorProto,
        .inputs = try graph_inputs.toOwnedSlice(), // []*ValueInfoProto,
        .outputs = try graph_outputs.toOwnedSlice(), // []*ValueInfoProto,
        .value_info = try graph_value_infos.toOwnedSlice(), // []*ValueInfoProto,
    };

    return onnx.ModelProto{
        .ir_version = onnx.Version.IR_VERSION,
        .producer_name = null,
        .producer_version = null,
        .domain = null,
        .model_version = null,
        .doc_string = null,
        .graph = graph_ptr,
    };
}
