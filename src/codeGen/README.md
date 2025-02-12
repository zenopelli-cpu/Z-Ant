# Zant Code Generation

The purpose of Zant code generation is to create a zig file containing some methods that can be than exported into a static library.  

The Starting file is `codeGen_main.zig`, where the onnx model is parsed and passed to the code generator.

## Structures
### ReadyNode
- nodeProto: pointer to an onnx node. I decide to keep it so tho have a reference to the onnx correspondant.  
- inputs: list of ReadyTensor representing the inputs of the node  
- outputs: list of ReadyTensor representing the outputs of the node  
- ready: this flag is set to true if the node has all the inputs and all the outputs are set as ready  

### ReadyTensor
- name: name of the tensor as presented in the onnx file (I sanitize the name only when writing it on the generated file, it is the last step)  
- 