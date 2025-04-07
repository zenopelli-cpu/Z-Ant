# TensorMath package

TensorMath package includes all the class necessary to perform matematical operation. When adding a new method, choose the proper `*_math.zig` file, or if you can't fid a good one create it by yourself. 

## op VS op_lean
The difference between those methodsis that `op_lean` doesn't perform any checks and always return void. The lean version of the operation is used only for MCU inference with ONNX format.   
Each  `*op_something.zig` implements both the standard version and the lean one.

**OSS**: lean_tensor_math is WIP

## op_something VS lib_something
Op_ contains only one method (operation) while lib_ groups a choort of methods. If you are implementing a complex functionality that doesn't match any of the lib_ crate a op_yourMethod.


# How to add a math operator
For each mathematical operation you want to implement you must write 3 pub methods :  
- operator() 
- operator_lean()
- get_operator_output_shape()  

They must have the following structure:

Standard version of the function, used for unit tests, returns a Tensor.
```zig
pub fn operator(input: Tensor(T), attributes...) !Tensor(T) {

    // checks
    ...

    // compute output
    const output_shape = get_operator_output_shape(...);
    var output_tensor = try Tensor(T).fromShape(allocator, output_shape);


    try operator_lean(input, output, attributes...);

}
```
The Lean version of the function, used during NN output prediction, returns void, no Tensor allocation inside the method.
```Zig
pub fn operator_lean(input: Tensor(T), output: Tensor(T),  attributes...) !void {
    // Actual computation of the output.
    // Dynamic allocation is forbidden here!
    // At this point you must assume that all the shapes are correct
    ...
}
```
A method to compute the shape of the output tensor. This must be the **ONLY** place in the code where the output shape is computed, avoid boilerplate code at all costs.
```Zig
pub fn get_operator_output_shape(...) ![]usize {
    // Given the args computes the shape of the output
    ...
}
```