# TensorMath package

TensorMath package includes all the class necessary to perform matematical operation. When adding a new method, choose the proper `*_math.zig` file, or if you can't fid a good one create it by yourself. 

## tensor_math.zig VS lean_tensor_math.zig
The difference between those import files (we can call them "libraries") is that `lean_tensor_math` doesn't perform any checks and always return void. The lean version of the library is used only for MCU inference with ONNX format.   
Each  `*_math.zig` implements both the standard version and the lean one.

**OSS**: lean_tensor_math is WIP

## op_something VS lib_something
Op_ contains only one method (operation) while lib_ groups a choort of methods. If you are implementing a complex functionality that doesn't match any of the lib_ crate a op_yourMethod.

