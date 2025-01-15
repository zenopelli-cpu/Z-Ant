# TensorMath package

TensorMath package includes all the class necessary to perform matematical operation. When adding a new method, choose the proper `*_math.zig` file, or if you can't fid a good one create it by yourself. 

## tensor_math.zig VS lean_tensor_math.zig
The difference between those import files (we can call them "libraries") is the usage of `Tensor` or `LeanTensor`. The lean version of the library is used only for MCU inference with ONNX format.   
Each  `*_math.zig` implements both the standard version and the lean one.

**OSS**: lean_tensor_math is WIP
