//! in this file you will find everything you need to fuse two or more operation
//! into one.
//!
//!  More specifically for this file:
//!     Conv -> BatchNormalization -> Relu
//!  is converted into:
//!     fused_Conv_BatchNormalization_Relu

const std = @import("std");
const zant = @import("zant");

const IR_zant = @import("../IR_zant.zig");

const fused_operators = IR_zant.fused_operators;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;

pub fn fuse_conv_batchNormlaization_relu(fusion_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
    return NodeZant.init_fused_node(
        fusion_list,
        fused_operators.Fused_Conv_BatchNormalization_Relu.init_fused_op,
        null,
        null,
    );
}
