- `globals.setGlobalAttributes()`

 

    - parse the input shape from codegen_options  
 

    - PROBLEM: just assign one network Input  
 

    - initialize the networkOutput  PROBLEM: what if we don't have the onnx output?
 

    - if present use the parsedInputshape from codegen_options else keep the onnx one PROBLEM: what if onnx do not have one?
 

    - `populateReadyTensorHashMap()`
 

        - adding initializers to tensorHashMap
 

        - adding all the nodes inputs and outputs to tensorHashMap OSS: the network output will be created as a .LINK
 

    - `populateReadyGraph()`
 

        - Creates a graph representation with all nodes in a ready-to-compute state, each node has a list for all the inputs and all the uputs, remember, everything is saved on the HasMap
 

        - `compute_output_shape()`
 

- `codeGen.skeleton.writeZigFile()`
 

    - writes the skeleton for the codegen file: `write_libraries(), write_logFunction(),  write_FBA(), write_type_T()...`
 

    - `write_parameters()`
 

        - Writes the Zig code required to initialize all tensor initializers in the ONNX model
 

    - `writePredict()`
 

        - `write_outputsInitialization()`
 

            - for each element of readyGraph, for the output of each node, codegen the relative Tensor
 

        - `write_outputsResetMethod(), write_checks(), write_predictInitialization() `
 

        - `write_graphSerialization()` IMPORTANT
 

            - while not reached the last nod of the network
 

                -  retrive the list of computable nodes with `getComputableNodes()`
 

                - save the last of the computable nodes
 

                - codegen the operation on the file with `writeOperation()` and set the output of that node to "ready". Remember, the ReadyTensor is set to "ready" and the only copy of that object is present inside `tensorHashMap`.
 


 
