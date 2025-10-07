---
name: 'Node Fusion Template'
about: Related to Zant Math.
title: ''
labels: ''
assignees: ''

---

## Current Problem
Describe here your issue

# Implementation 

## Interface 
Write the `fused_op.zig` file inside `src/IR_zant/fusion/op_union/fused_operators`.   
Each fused node must contain the following methods:
    
- `pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Op` for the Node initialization.
- `fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant)` contains the pattern matching logic. Depending on how you implement it the **root_node** may change, be clear on your implementation logic. It returns a list contating the pointers to the nodes you want to fuse.
- `fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant ` Given the list of nodes you want to fuse it creates a NodeZant representig the new fused node.
- `fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void` takes the nodes to fuse nodes and sobstitutes them with the node representing the fusion. Pay attention to adjust the output of the parents nodes to point to the correct node.

## Strategy
You can represent a fused node in two different ways:  
1) By fused nodes collection: so if you want to fuse Op1 and Op2 into New_fused_op, it will have the following structure:
``` 
    pub const New_fused_op = struct {
        op_name: []const u8,
        op_1: operators.Op1, // Use the actual Conv type
        op_2: operators.Op2, // Use the actual Relu type
        ...
    }
```

2) By attribute, you save all the needed attributes by copying them from the fused nodes:
```
    pub const New_fused_op = struct {
        op_name: []const u8,
        op_1_weight: *TensorZant, //op_1 attribute
        op_1_bias: *TensorZant, //op_1 attribute
        op_1_numbers: *TensorZant, //op_1 attribute
        op_2_weight: *TensorZant, //op_2 attribute
        op_2_bias: *TensorZant, //op_2 attribute
        op_2_number: int, //op_2 attribute
        ...
    }
```
 
Strategy **1** is recommended.

After this add your patten to `src/IR_zant/fusion/pattern_collection.zig`.

## Next
Follow [this](../../src/IR_zant/op_union/HOW_TO_ADD_MATHEMATICAL_OPERATIONS.md) guide to implement the math logic.

## Testing
To test the new fusion pattern it is required to create a custom-made ONNX model containing the pattern and follow the minimal workflow shown in **docs/ZANT_CLI.md**.
