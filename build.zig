const std = @import("std");

/// Entry point for the build system.
/// This function defines how to build the project by specifying various modules and their dependencies.
/// @param b - The build context, which provides utilities for configuring the build process.
pub fn build(b: *std.Build) void {
    const build_options = b.addOptions();
    build_options.addOption(bool, "trace_allocator", b.option(bool, "trace_allocator", "Use a tracing allocator") orelse true);
    build_options.addOption([]const u8, "allocator", (b.option([]const u8, "allocator", "Allocator to use") orelse "raw_c_allocator"));

    // Set target options, such as architecture and OS.
    const target = b.standardTargetOptions(.{});

    // Set optimization level (debug, release, etc.).
    const optimize = b.standardOptimizeOption(.{});

    //************************************************MODULE CREATION************************************************

    // Create modules from the source files in the `src/Core/Tensor/` directory.
    const tensor_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/tensor.zig") });

    // Create modules from the source files in the `src/Core/Tensor/TensorMath` directory.
    const tensor_math_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/TensorMath/tensor_math_standard.zig") });
    // const tensor_math_lean_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/TensorMath/tensor_math_lean.zig") });

    // Create modules from the source files in the `src/Model/` directory.
    const loss_mod = b.createModule(.{ .root_source_file = b.path("src/Model/lossFunction.zig") });
    const activation_mod = b.createModule(.{ .root_source_file = b.path("src/Model/activation_function.zig") });
    const model_mod = b.createModule(.{ .root_source_file = b.path("src/Model/model.zig") });
    const layer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/layer.zig") });
    const optim_mod = b.createModule(.{ .root_source_file = b.path("src/Model/optim.zig") });

    // Create modules from the source files in the `src/Model/Layers` directory.
    const denseLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/denseLayer.zig") });
    const activationLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/activationLayer.zig") });
    const convLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/convLayer.zig") });
    const flattenLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/flattenLayer.zig") });
    const poolingLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/poolingLayer.zig") });
    const batchNormLayer_mod = b.createModule(.{ .root_source_file = b.path("src/Model/Layers/batchNormLayer.zig") });

    // onnx module
    const onnx_mod = b.createModule(.{ .root_source_file = b.path("src/onnx/onnx.zig") });

    // code generation module
    const codegen_mod = b.createModule(.{ .root_source_file = b.path("src/codeGen/codeGen_skeleton.zig") });

    // Create modules from the source files in the `src/DataHandler/` directory.
    const dataloader_mod = b.createModule(.{ .root_source_file = b.path("src/DataHandler/dataLoader.zig") });
    const dataProcessor_mod = b.createModule(.{ .root_source_file = b.path("src/DataHandler/dataProcessor.zig") });
    const trainer_mod = b.createModule(.{ .root_source_file = b.path("src/DataHandler/trainer.zig") });

    // Create modules from the source files in the `src/utils/` directory.
    const typeConv_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/typeConverter.zig") });
    const errorHandler_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/errorHandler.zig") });
    const modelImportExport_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/model_import_export.zig") });
    const allocator_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/allocator.zig") });
    allocator_mod.addOptions("build_options", build_options);

    //************************************************MODEL DEPENDENCIES************************************************

    // Add necessary imports for the model module.
    model_mod.addImport("tensor", tensor_mod);
    model_mod.addImport("layer", layer_mod);
    model_mod.addImport("optim", optim_mod); // Do not remove duplicate
    model_mod.addImport("loss", loss_mod);
    model_mod.addImport("typeC", typeConv_mod);
    model_mod.addImport("dataloader", dataloader_mod);
    model_mod.addImport("tensor_m", tensor_math_mod);
    model_mod.addImport("dataprocessor", dataProcessor_mod);
    model_mod.addImport("activation_function", activation_mod);

    // ************************************************LAYER DEPENDENCIES************************************************

    // Add necessary imports for the layers module.
    layer_mod.addImport("tensor", tensor_mod);
    layer_mod.addImport("activation_function", activation_mod);
    layer_mod.addImport("tensor_m", tensor_math_mod);
    layer_mod.addImport("errorHandler", errorHandler_mod);
    layer_mod.addImport("pkgAllocator", allocator_mod);

    // All layers are imported so that the layer module can be used as a layer library by other modules.
    // New layers should be added here.
    layer_mod.addImport("activationLayer", activationLayer_mod);
    layer_mod.addImport("batchNormLayer", batchNormLayer_mod);
    layer_mod.addImport("convLayer", convLayer_mod);
    layer_mod.addImport("denseLayer", denseLayer_mod);
    layer_mod.addImport("flattenLayer", flattenLayer_mod);
    layer_mod.addImport("poolingLayer", poolingLayer_mod);

    // ************************************************DENSELAYER DEPENDENCIES************************************************

    // Add necessary imports for the denselayers module.
    denseLayer_mod.addImport("tensor", tensor_mod);
    denseLayer_mod.addImport("tensor_m", tensor_math_mod);
    denseLayer_mod.addImport("Layer", layer_mod);
    denseLayer_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************CONVLAYER DEPENDENCIES************************************************
    convLayer_mod.addImport("Tensor", tensor_mod);
    convLayer_mod.addImport("tensor_m", tensor_math_mod);
    convLayer_mod.addImport("Layer", layer_mod);
    convLayer_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************FLATTENLAYER DEPENDENCIES************************************************

    flattenLayer_mod.addImport("Tensor", tensor_mod);
    flattenLayer_mod.addImport("tensor_m", tensor_math_mod);
    flattenLayer_mod.addImport("Layer", layer_mod);
    flattenLayer_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************POOLINGLAYER DEPENDENCIES************************************************
    poolingLayer_mod.addImport("Tensor", tensor_mod);
    poolingLayer_mod.addImport("tensor_m", tensor_math_mod);
    poolingLayer_mod.addImport("Layer", layer_mod);
    poolingLayer_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************ACTIVATIONLAYER DEPENDENCIES************************************************

    // Add necessary imports for the activationlayers module.
    activationLayer_mod.addImport("tensor", tensor_mod);
    activationLayer_mod.addImport("tensor_m", tensor_math_mod);
    activationLayer_mod.addImport("Layer", layer_mod);
    activationLayer_mod.addImport("activation_function", activation_mod);
    activationLayer_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************BATCHNORMLAYER DEPENDENCIES************************************************
    batchNormLayer_mod.addImport("Tensor", tensor_mod);
    batchNormLayer_mod.addImport("tensor_m", tensor_math_mod);
    batchNormLayer_mod.addImport("Layer", layer_mod);
    batchNormLayer_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************DATA LOADER DEPENDENCIES************************************************

    // Add necessary imports for the data loader module.
    dataloader_mod.addImport("tensor", tensor_mod);

    // ************************************************DATA PROCESSOR DEPENDENCIES************************************************

    // Add necessary imports for the data processor module.
    dataProcessor_mod.addImport("tensor", tensor_mod);

    // ************************************************TRAINER DEPENDENCIES************************************************

    // Add necessary imports for the trainer module.
    trainer_mod.addImport("tensor", tensor_mod);
    trainer_mod.addImport("tensor_m", tensor_math_mod);
    trainer_mod.addImport("model", model_mod);
    trainer_mod.addImport("loss", loss_mod);
    trainer_mod.addImport("optim", optim_mod);
    trainer_mod.addImport("dataloader", dataloader_mod);
    trainer_mod.addImport("dataprocessor", dataProcessor_mod);
    trainer_mod.addImport("layer", layer_mod);

    // ************************************************TENSOR DEPENDENCIES************************************************

    // Add necessary imports for the tensor module.
    tensor_mod.addImport("tensor_m", tensor_math_mod);
    tensor_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************TENSOR MATH DEPENDENCIES************************************************
    // Add necessary imports for the tensor math module.
    // OSS: Do not import file from the same folder ./TensorMath!! Directly use @import()"filename.zig")
    // Import in tensor_math_mod all the modules needed for /TensorMath files
    tensor_math_mod.addImport("tensor", tensor_mod);
    tensor_math_mod.addImport("typeC", typeConv_mod);
    tensor_math_mod.addImport("errorHandler", errorHandler_mod);
    tensor_math_mod.addImport("layer", layer_mod);
    tensor_math_mod.addImport("pkgAllocator", allocator_mod);

    // ************************************************ACTIVATION DEPENDENCIES************************************************

    // Add necessary imports for the activation module.
    activation_mod.addImport("tensor", tensor_mod);
    activation_mod.addImport("errorHandler", errorHandler_mod);
    activation_mod.addImport("pkgAllocator", allocator_mod);
    activation_mod.addImport("tensor_m", tensor_math_mod);

    // ************************************************LOSS DEPENDENCIES************************************************

    // Add necessary imports for the loss function module.
    loss_mod.addImport("tensor", tensor_mod);
    loss_mod.addImport("tensor_m", tensor_math_mod);
    loss_mod.addImport("typeC", typeConv_mod);
    loss_mod.addImport("errorHandler", errorHandler_mod);
    loss_mod.addImport("pkgAllocator", allocator_mod);

    // ************************************************OPTIMIZER DEPENDENCIES************************************************

    // Add necessary imports for the optimizer module.
    optim_mod.addImport("tensor", tensor_mod);
    optim_mod.addImport("model", model_mod);
    optim_mod.addImport("layer", layer_mod);
    optim_mod.addImport("errorHandler", errorHandler_mod);

    // ************************************************IMPORT/EXPORT DEPENDENCIES************************************************

    // Add necessary imports for the import/export module.
    modelImportExport_mod.addImport("tensor", tensor_mod);
    modelImportExport_mod.addImport("layer", layer_mod);
    modelImportExport_mod.addImport("model", model_mod);
    modelImportExport_mod.addImport("errorHandler", errorHandler_mod);
    modelImportExport_mod.addImport("activation_function", activation_mod);

    // ************************************************CODEGEN DEPENDENCIES************************************************
    codegen_mod.addImport("tensor", tensor_mod);
    codegen_mod.addImport("onnx", onnx_mod);
    codegen_mod.addImport("pkgAllocator", allocator_mod);

    // ************************************************MAIN EXECUTABLE************************************************

    // Define the main executable with target architecture and optimization settings.
    const exe = b.addExecutable(.{
        .name = "Main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.linkLibC();

    // ************************************************MAIN DEPENDENCIES************************************************

    // Add necessary imports for the main executable.
    exe.root_module.addImport("tensor", tensor_mod);
    exe.root_module.addImport("model", model_mod);
    exe.root_module.addImport("layer", layer_mod);
    exe.root_module.addImport("dataloader", dataloader_mod);
    exe.root_module.addImport("dataprocessor", dataProcessor_mod);
    exe.root_module.addImport("activation_function", activation_mod);
    exe.root_module.addImport("loss", loss_mod);
    exe.root_module.addImport("trainer", trainer_mod);
    exe.root_module.addImport("pkgAllocator", allocator_mod);
    exe.root_module.addImport("model_import_export", modelImportExport_mod);

    // Install the executable.
    b.installArtifact(exe);

    // Define the run command for the main executable.
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    // ************************************************CODEGEN EXECUTABLE************************************************

    // Define the main executable with target architecture and optimization settings.
    const codeGen_exe = b.addExecutable(.{
        .name = "Codegen",
        .root_source_file = b.path("src/codeGen/codeGen_main.zig"),
        .target = target,
        .optimize = optimize,
    });

    codeGen_exe.linkLibC();

    // ************************************************CODEGEN DEPENDENCIES************************************************

    // Add necessary imports for the executable.
    codeGen_exe.root_module.addImport("onnx", onnx_mod);
    codeGen_exe.root_module.addImport("codeGen", codegen_mod);

    // Install the executable.
    b.installArtifact(codeGen_exe);

    // Define the run command for the main executable.
    const codegen_cmd = b.addRunArtifact(codeGen_exe);
    if (b.args) |args| {
        codegen_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const codegen_step = b.step("codegen", " code generation");
    codegen_step.dependOn(&codegen_cmd.step);

    //************************************************UNIT TESTS************************************************

    // Define unified tests for the project.
    const unit_tests = b.addTest(.{
        .name = "lib",
        .root_source_file = b.path("tests/test_lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Define test options
    const test_options = b.addOptions();
    test_options.addOption(bool, "heavy", b.option(bool, "heavy", "Run heavy tests") orelse false);
    unit_tests.root_module.addOptions("test_options", test_options);

    //************************************************UNIT TEST DEPENDENCIES************************************************

    // Add necessary imports for the unit test module.
    unit_tests.root_module.addImport("tensor", tensor_mod);
    unit_tests.root_module.addImport("model", model_mod);
    unit_tests.root_module.addImport("layer", layer_mod);
    unit_tests.root_module.addImport("optim", optim_mod);
    unit_tests.root_module.addImport("loss", loss_mod);
    unit_tests.root_module.addImport("tensor_m", tensor_math_mod);
    unit_tests.root_module.addImport("activation_function", activation_mod);
    unit_tests.root_module.addImport("dataloader", dataloader_mod);
    unit_tests.root_module.addImport("dataprocessor", dataProcessor_mod);
    unit_tests.root_module.addImport("trainer", trainer_mod);
    unit_tests.root_module.addImport("typeConverter", typeConv_mod);
    unit_tests.root_module.addImport("errorHandler", errorHandler_mod);
    unit_tests.root_module.addImport("model_import_export", modelImportExport_mod);
    unit_tests.root_module.addImport("pkgAllocator", allocator_mod);

    unit_tests.linkLibC();

    // Add a build step to run all unit tests.
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
