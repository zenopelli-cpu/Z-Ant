✅ : complete  
🔶 : WIP  
🔴 : missing  
  
onnx reference : \[onnx_name\]\(URL to [ONNX docs](https://onnx.ai/onnx/operators/index.html)\)  
tensor math : \[ fileName.zig \]\( path to the method \)  
tensor math tests: ✅, 🔶, 🔴  
codegen : ✅, 🔶, 🔴. You can find all the `write_op()` [here](../src/CodeGen/math_handler.zig)  
oneOp model generator: ✅ if the oneOpModel is created, remember to add the onnx name inside [available_op](all_link_here)



| math op name | onnx reference | tensor math | tensor math tests | codegen | oneOp model generator (.py) | notes |
| :------------: | :------------: | :---------: | :-----------: | :-------: | :--------: | :------------- |
| convolution | [Conv](https://onnx.ai/onnx/operators/onnx__Conv.html) | [op_convolution.zig](../src/Core/Tensor/TensorMath/op_convolution.zig) | ✅ | ✅ | ✅ |
| gemm | [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html) | [op_gemm](../src/Core/Tensor/TensorMath/op_gemm.zig) | ✅ | ✅ | ✅ |
|Div| [Div](https://onnx.ai/onnx/operators/onnx__Div.html) |[op_div](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_division.zig) | ✅ | ✅ | ✅ |
|Add| [Add](https://onnx.ai/onnx/operators/onnx__Add.html) | [op_add](../src/Core/Tensor/TensorMath/lib_elementWise_math/op_addition.zig) | ✅ | ✅ | ✅ |
|Concat| [Concat](https://onnx.ai/onnx/operators/onnx__Concat.html)| [op_concat](../src/Core/Tensor/TensorMath/lib_shape_math/op_concatenate.zig) | ✅ | ✅ | ✅ |

