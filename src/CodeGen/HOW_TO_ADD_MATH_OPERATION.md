# How to add mathematical operations to codegen 
Read this document before adding a math operation to `math_handler.zig` .  
First lets see briefly how does the codegen works:  
- given a onnx model called `model.onnx` the parser create a Zant representation of the model, you can find more information inside `src/onnx` 
- this intermediate representation is than used to reconstruct the graph. In the graph each node trepresent a mathematical operation and each node can have **multiple** inputs and multiple outputs.
The inputs can be: **mandatory** or **optional**. Taking care of what type of input you are working with is crucial.
- given the intermediate representation we can now start to write the file that, when compiled, will exectute the graph operations.   
## How do we codegen the predict() ?

First of all you must understand the global structures we use:
#### ReadyTensor
attributes:

    name: []const u8,
    ready: bool,
    shape: []const i64,
    tensorProto: ?*TensorProto,
    tag: TensorTag,
where TensorTag:
```C++
enum {
    INITIALIZER,
    CONSTANT,
    INPUT,
    OUTPUT,
    LINK, 
};
```

#### ReadyNode
attributes:

    nodeProto: *NodeProto,
    inputs: std.ArrayList(*ReadyTensor),
    outputs: std.ArrayList(*ReadyTensor),
    ready: bool,
**OSS**: `inputs` and `outputs` are pointers to ReadyTensor

#### readyGraph
A list of ReadyNodes, is the representatio of our graph.

#### tensorHashMap
tensorHashMap is an hash map where the key is the non-sanitized name of the Tensor and the value is the ReadyTensor. It is the **most important structure since it is the only one containing ReadyTensor instances!**, all the other structures contains pointers to Values of this hash map. We can assume each key is unique by onnx naming definition. 

---

Now that we have a general idea of how the inforamtions are saved and managed, we can start explaning how to add mathematical operations to the codegen.
The method aiming to write all the mathematical operations is `write_math_op()` inside `math_handler.zig`. Depending on what type of ReadyNode.nodeProto.op_type we call the correspondant `write_op_type()`. 
    
Let's start with and **example**:  
Assume that in the [onnx standard](https://onnx.ai/onnx/operators/index.html) exists the `spaghetti(A, B, ?C, D, E)` operation where:  
Inputs:

    A (heterogeneous) - T
    B (heterogeneous) - T
    C - FLOAT (optional, heterogeneous)
    D - FLOAT (default is '1.0'):
Outputs:  

    E (heterogeneous) - T 

When looking at `ReadyNode.inputs` we can have different types of ReadyTensors and we must generate the code depending on that.
If a tensor is "requested" is for sure present in the `ReadyNode.inputs` and it is tagged as .LINK or .INITIALIZER in the same position as declare in the onnx standard, but what if it is "optional" and is is followed by another non-optional input? The `ReadyNode.inputs.items.blen == 3` where:  
`ReadyNode.inputs.items[0]` is A  
`ReadyNode.inputs.items[1]` is B  
`ReadyNode.inputs.items[2]` is D   
so if ignore the fact that C may be not present in `ReadyNode.inputs` my codegen would produce something like this:
```C++
//                A           B          C        D
spaghetti_lean (&tensor_A, &tensor_B, &tensor_D, ERROR?!?!?!);
```
To avoid this problem every time I'm working with optional parameters I have to check if they are present in the **tensorHashMap** or if the inputs list is long enought when the Tensor is the last one in the list(happens often), if not, set them to "null". 
  
      
  
Every time a new math operation is added remember to add it into `docs/MATH_TABLE`!! 



