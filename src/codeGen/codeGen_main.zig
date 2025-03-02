const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const allocator = zant.utils.allocator.allocator;
const codeGen = @import("codeGen_skeleton.zig");
const codeGen_utils = @import("codeGen_utils.zig");
const codeGen_init = @import("codeGen_parameters.zig");
const codeGen_mathHandl = @import("codeGen_math_handler.zig");
const codeGen_predict = @import("codeGen_predict.zig");
const codeGen_tests = @import("codeGen_tests.zig");

const codegen_options = @import("codegen_options");
const globals = @import("globals.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    const model_name = "mnist-8";
    // Format model path according to model_name
    const model_path = "datasets/models/" ++ model_name ++ "/" ++ model_name ++ ".onnx";
    var model = try onnx.parseFromFile(gpa_allocator, model_path);
    defer model.deinit(gpa_allocator);

    //onnx.printStructure(&model);

    // Create the generated model directory if not present
    //const generated_path = "generated/" ++ model_name ++ "/";
    const generated_path = "src/codeGen/";
    try std.fs.cwd().makePath(generated_path);

    // ONNX model parsing

    //create the hashMap
    try globals.populateReadyTensorHashMap(model);

    // model.print();

    //DEBUG
    //utils.printTensorHashMap(tensorHashMap);

    //DEBUG
    //try utils.printOperations(model.graph.?);

    //create the ReadyGraph
    try globals.populateReadyGraph(model);

    //DEBUG
    //try utils.printNodeList(readyGraph);

    //////////////////////////////////////////

    // Create the code for the model
    try codeGen.writeZigFile(model_name, generated_path, model);

    // Test the generated code
    try codeGen_tests.writeTestFile(model_name, generated_path, model);
}
