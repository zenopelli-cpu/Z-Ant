const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;

const IR_zant = @import("IR_zant");
const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const allocator = zant.utils.allocator.allocator;
const codeGen = @import("codegen_v2.zig");
const codeGen_utils = codeGen.utils;
const codeGen_init = codeGen.parameters;
const codeGen_predict = codeGen.predict;
const codeGen_tests = codeGen.tests;
const codegen_options = codeGen.codegen_options;

pub fn main_v2(model: *ModelOnnx) !void {
    const model_name = codegen_options.model;

    var graphZant: IR_zant.GraphZant = try IR_zant.init(model);
    defer graphZant.deinit();

    model.print();

    // Create the generated model directory if not present
    const generated_path = codegen_options.generated_path;
    //const generated_path = "src/codeGen/";
    try std.fs.cwd().makePath(generated_path);

    //DEBUG
    //utils.printTensorHashMap(tensorHashMap);

    //DEBUG
    //try utils.printOperations(model.graph.?);

    //DEBUG
    //try utils.printNodeList(readyGraph);

    //////////////////////////////////////////

    // Create the code for the model
    try codeGen.skeleton.writeZigFile(model_name, generated_path, graphZant.nodes, true);

    // Test the generated code
    try codeGen_tests.writeTestFile(model_name, generated_path);

    //PRINTING DETAILS OF THE MODEL
    try onnx.printModelDetails(&model);
}
