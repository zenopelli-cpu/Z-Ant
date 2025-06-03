const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const IR_graph = zant.IR_graph;
const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const allocator = zant.utils.allocator.allocator;
const codeGen = @import("codegen.zig");
const codeGen_utils = codeGen.utils;
const codeGen_init = codeGen.parameters;
const codeGen_mathHandl = codeGen.math_handler;
const codeGen_predict = codeGen.predict;
const codeGen_tests = codeGen.tests;

const codegen_options = @import("codegen_options");
const globals = codeGen.globals;

pub fn main() !void {
    const model_name = codegen_options.model;
    const model_path = codegen_options.model_path;

    var model: onnx.ModelProto = try onnx.parseFromFile(allocator, model_path);
    defer model.deinit(allocator);

    var graphZant: IR_graph.GraphZant = try IR_graph.init(&model);
    defer graphZant.deinit();

    model.print();

    // Create the generated model directory if not present
    const generated_path = codegen_options.generated_path;
    //const generated_path = "src/codeGen/";
    try std.fs.cwd().makePath(generated_path);

    // ONNX model parsing
    try globals.setGlobalAttributes(model);

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
