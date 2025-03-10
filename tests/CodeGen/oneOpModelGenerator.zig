//! The aim of this file is that, given a onnx oprator (https://onnx.ai/onnx/operators/),
//! it returns the onnx.ModelProto containing only one node for the operation

const std = @import("std");
const zant = @import("zant");

const onnx = zant.onnx;
const codeGen = zant.codeGen;

//------------operation structs
//Add
const AddStruct = struct {
    name: []const u8,
    inputShape: []i64,
    //other...
};

//Concat
const ConcatStruct = struct {
    name: []const u8,
    inputShape: []i64,
    //other...
};

//Conv
const ConvStruct = struct {
    name: []const u8,
    inputShape: []i64,
    //other...
};

//------------union of all the operation struct
const OpStruct = union(enum) {
    Add: AddStruct,
    Concat: ConcatStruct,
    Conv: ConvStruct,
};

pub fn oneOpModel(
    op_struct: OpStruct,
) !onnx.ModelProto {
    var modelProto = onnx.ModelProto{
        .ir_version = onnx.Version.IR_VERSION,
        .producer_name = null,
        .producer_version = null,
        .domain = null,
        .model_version = null,
        .doc_string = null,
        .graph = &onnx.GraphProto{ //?*GraphProto,
            .name = null, //?[]const u8,
            .nodes = undefined, // []*NodeProto,
            .initializers = undefined, // []*TensorProto,
            .inputs = undefined, // []*ValueInfoProto,
            .outputs = undefined, // []*ValueInfoProto,
            .value_info = undefined, // []*ValueInfoProto,
        },
    };

    switch (op_struct) {
        .Add => {
            //create the correct node and complete the modelProto.graph

            _ = &modelProto;
        },
        .Concat => {
            //create the correct node and complete the modelProto.graph

            _ = &modelProto;
        },
        .Conv => {
            //create the correct node and complete the modelProto.graph

            _ = &modelProto;
        },
    }

    return modelProto;
}
