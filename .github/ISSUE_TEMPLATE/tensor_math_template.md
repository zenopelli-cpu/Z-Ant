---
name: 'Tensor Math Template'
about: Related to TensorMath functions.
title: ''
labels: ''
assignees: ''

---

## Current Problem
Describe here your problem

## Rules
### ONNX standard
Every math method must respect the onnx representation.
You can find all the necessary details about onnx operators [here](https://onnx.ai/onnx/operators/index.html) !  

### Standard & lean format

The "standard version" of a method is in the format:

```  
    pub fn methodName (args) !Tensor(T) {
       ...
       outputShape = compute output shape...
       outputTensor = compute output tensor...
       ...
       methodName_lean(input, args..., output)

       return outputTensor;
    }
```

It is suggested to write a `get_methodName_oputput_shape()` to compute the shape of the output so that the same method can be called during the generation of the readyGraph.

The "lean version" of a method is in the format:

```  
    pub inline fn methodName_lean (args) void {
       ...
       ... actual computation of the output
    }
```

"lean" and "standard" are **mandatory**.
The lean version is called during the codegen and the standard version is used in testings.