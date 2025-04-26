const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;

//--- protos
const GraphProto = zant.onnx.GraphProto;
const NodeProto = zant.onnx.NodeProto;

pub const GraphZant = @import("graphZant.zig").GraphZant;
pub const NodeZant = @import("nodeZant.zig").NodeZant;

pub fn init(modelProto: ModelProto) *GraphZant {
    const graphProto = modelProto.graph;

    //given the onnx model iterate over all nodes
    for (graphProto.nodes) |nodeProto| {

        //create all the new NodeZant

        //for each node search for the child_node

    }
}
