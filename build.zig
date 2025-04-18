const std = @import("std");

/// Entry point for the build system.
/// This function defines how to build the project by specifying various modules and their dependencies.
/// @param b - The build context, which provides utilities for configuring the build process.
pub fn build(b: *std.Build) void {
    const build_options = b.addOptions();
    build_options.addOption(bool, "trace_allocator", b.option(bool, "trace_allocator", "Use a tracing allocator") orelse true);
    build_options.addOption([]const u8, "allocator", (b.option([]const u8, "allocator", "Allocator to use") orelse "raw_c_allocator"));

    // Get target and CPU options from command line or use defaults
    const target_str = b.option([]const u8, "target", "Target architecture (e.g., thumb-freestanding)") orelse "native";
    const cpu_str = b.option([]const u8, "cpu", "CPU model (e.g., cortex_m33)");

    const target_query = std.Target.Query.parse(.{
        .arch_os_abi = target_str,
        .cpu_features = cpu_str,
    }) catch |err| {
        std.debug.print("Error parsing target: {}\n", .{err});
        return;
    };

    const target = b.resolveTargetQuery(target_query);
    const optimize = b.standardOptimizeOption(.{});

    // -------------------- Modules creation

    const zant_mod = b.createModule(.{ .root_source_file = b.path("src/zant.zig") });
    zant_mod.addOptions("build_options", build_options);

    //************************************************UNIT TESTS************************************************
    const codeGen_mod = b.createModule(.{ .root_source_file = b.path("src/CodeGen/codegen.zig") });
    codeGen_mod.addImport("zant", zant_mod);

    // Define unified tests for the project.
    const unit_tests = b.addTest(.{
        .name = "test_lib",
        .root_source_file = b.path("tests/test_lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    unit_tests.root_module.addImport("zant", zant_mod);
    unit_tests.root_module.addImport("codegen", codeGen_mod);

    unit_tests.linkLibC();

    // Add a build step to run all unit tests.
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ************************************************CODEGEN EXECUTABLE************************************************

    // Define the main executable with target architecture and optimization settings.
    const codeGen_exe = b.addExecutable(.{
        .name = "Codegen",
        .root_source_file = b.path("src/CodeGen/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    codeGen_exe.linkLibC();

    // Add necessary imports for the executable.
    codeGen_exe.root_module.addImport("zant", zant_mod);

    // Name and path of the model
    const model_name_option = b.option([]const u8, "model", "Model name") orelse "mnist-8";
    const model_path_option = b.option([]const u8, "model_path", "Model path") orelse std.fmt.allocPrint(b.allocator, "datasets/models/{s}/{s}.onnx", .{ model_name_option, model_name_option }) catch |err| {
        std.debug.print("Error allocating model path: {}\n", .{err});
        return;
    };

    // Generated path
    var generated_path_option = b.option([]const u8, "generated_path", "Generated path") orelse "";
    if (generated_path_option.len == 0) {
        generated_path_option = std.fmt.allocPrint(b.allocator, "generated/{s}/", .{model_name_option}) catch |err| {
            std.debug.print("Error allocating generated path: {}\n", .{err});
            return;
        };
    } else {
        if (!std.mem.endsWith(u8, generated_path_option, "/")) {
            generated_path_option = std.fmt.allocPrint(b.allocator, "{s}/", .{generated_path_option}) catch |err| {
                std.debug.print("Error normalizing path: {}\n", .{err});
                return;
            };
        }
        generated_path_option = std.fmt.allocPrint(b.allocator, "{s}{s}/", .{ generated_path_option, model_name_option }) catch |err| {
            std.debug.print("Error allocating generated path: {}\n", .{err});
            return;
        };
    }

    // Define codegen options
    const codegen_options = b.addOptions(); // Model name option
    codegen_options.addOption([]const u8, "model", model_name_option);
    codegen_options.addOption([]const u8, "model_path", model_path_option);
    codegen_options.addOption([]const u8, "generated_path", generated_path_option);
    codegen_options.addOption([]const u8, "user_tests", b.option([]const u8, "user_tests", "User tests path") orelse "");
    codegen_options.addOption(bool, "log", b.option(bool, "log", "Run with log") orelse false);
    codegen_options.addOption([]const u8, "shape", b.option([]const u8, "shape", "Input shape") orelse "");
    codegen_options.addOption([]const u8, "type", b.option([]const u8, "type", "Input type") orelse "f32");
    codegen_options.addOption(bool, "comm", b.option(bool, "comm", "Codegen with comments") orelse false);
    codeGen_exe.root_module.addOptions("codegen_options", codegen_options);

    // Install the executable.
    b.installArtifact(codeGen_exe);

    // Define the run command for the main executable.
    const codegen_cmd = b.addRunArtifact(codeGen_exe);
    if (b.args) |args| {
        codegen_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const codegen_step = b.step("codegen", "code generation");
    codegen_step.dependOn(&codegen_cmd.step);

    // ************************************************ STATIC LIBRARY CREATION ************************************************

    const lib_model_path = std.fmt.allocPrint(b.allocator, "{s}lib_{s}.zig", .{ generated_path_option, model_name_option }) catch |err| {
        std.debug.print("Error allocating lib model path: {}\n", .{err});
        return;
    };

    const static_lib = b.addStaticLibrary(.{
        .name = "zant",
        .root_source_file = b.path(lib_model_path),
        .target = target,
        .optimize = .ReleaseSmall,
    });
    static_lib.linkLibC();
    static_lib.root_module.addImport("zant", zant_mod);
    static_lib.root_module.addImport("codegen", codeGen_mod);

    const install_lib_step = b.addInstallArtifact(static_lib, .{ .dest_dir = .{ .override = .{ .custom = model_name_option } } });
    const lib_step = b.step("lib", "Compile tensor_math static library");
    lib_step.dependOn(&install_lib_step.step);

    // Output path for the generated library
    var output_path_option = b.option([]const u8, "output_path", "Output path") orelse "";
    if (output_path_option.len != 0) {
        if (!std.mem.endsWith(u8, output_path_option, "/")) {
            output_path_option = std.fmt.allocPrint(b.allocator, "{s}/", .{output_path_option}) catch |err| {
                std.debug.print("Error normalizing path: {}\n", .{err});
                return;
            };
        }
        const old_path = std.fmt.allocPrint(b.allocator, "zig-out/{s}/", .{model_name_option}) catch |err| {
            std.debug.print("Error allocating old path: {}\n", .{err});
            return;
        };
        output_path_option = std.fmt.allocPrint(b.allocator, "{s}{s}/", .{ output_path_option, model_name_option }) catch |err| {
            std.debug.print("Error allocating output path: {}\n", .{err});
            return;
        };
        const move_step = b.addSystemCommand(&[_][]const u8{
            "mv",
            old_path,
            output_path_option,
        });
        move_step.step.dependOn(&install_lib_step.step);
        lib_step.dependOn(&move_step.step);
        move_step.step.dependOn(&install_lib_step.step);
        lib_step.dependOn(&move_step.step);
    }

    // ************************************************ GENERATED LIBRARY TESTS ************************************************

    // Add test for generated library
    const test_model_path = std.fmt.allocPrint(b.allocator, "{s}test_{s}.zig", .{ generated_path_option, model_name_option }) catch |err| {
        std.debug.print("Error allocating test model path: {}\n", .{err});
        return;
    };

    const test_generated_lib = b.addTest(.{
        .name = "test_generated_lib",
        .root_source_file = b.path(test_model_path),
        .target = target,
        .optimize = optimize,
    });

    test_generated_lib.root_module.addImport("zant", zant_mod);
    test_generated_lib.root_module.addImport("codegen", codeGen_mod);
    test_generated_lib.linkLibC();

    const run_test_generated_lib = b.addRunArtifact(test_generated_lib);
    const test_step_generated_lib = b.step("test-generated-lib", "Run generated library tests");
    test_step_generated_lib.dependOn(&run_test_generated_lib.step);

    // ************************************************ ONEOP CODEGEN ************************************************

    // Setup oneOp codegen

    const oneop_codegen_exe = b.addExecutable(.{
        .name = "oneop_codegen",
        .root_source_file = b.path("tests/CodeGen/oneOpModelGenerator.zig"),
        .target = target,
        .optimize = optimize,
    });

    oneop_codegen_exe.root_module.addImport("zant", zant_mod);
    codeGen_mod.addOptions("codegen_options", codegen_options);
    oneop_codegen_exe.root_module.addImport("codegen", codeGen_mod);
    oneop_codegen_exe.linkLibC();

    const run_oneop_codegen_exe = b.addRunArtifact(oneop_codegen_exe);
    const step_test_oneOp_codegen = b.step("test-codegen-gen", "Run generated library tests");
    step_test_oneOp_codegen.dependOn(&run_oneop_codegen_exe.step);

    // ************************************************

    //Setup test_all_oneOp

    const test_all_oneOp = b.addTest(.{
        .name = "test_all_oneOp",
        .root_source_file = b.path("generated/oneOpModels/test_oneop_models.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_all_oneOp.root_module.addImport("zant", zant_mod);
    codeGen_mod.addOptions("codegen_options", codegen_options);
    test_all_oneOp.root_module.addImport("codegen", codeGen_mod);
    test_all_oneOp.linkLibC();

    const run_test_all_oneOp = b.addRunArtifact(test_all_oneOp);

    // ************************************************
    // Setup test oneop
    // It will run
    // - run_oneop_codegen_exe
    // - run_test_all_oneOp

    const step_test_oneOp = b.step("test-codegen", "Run generated library tests");
    step_test_oneOp.dependOn(&run_test_all_oneOp.step);

    // ************************************************
    // Benchmark

    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const bench_options = b.addOptions();
    bench_options.addOption(bool, "full", b.option(bool, "full", "Choose whenever run full benchmark or not") orelse false);

    benchmark.root_module.addImport("zant", zant_mod);
    benchmark.root_module.addOptions("bench_options", bench_options);
    benchmark.linkLibC();

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);

    // ************************************************ ONNX PARSER TESTS ************************************************
    // Add test for generated library

    const test_onnx_parser = b.addTest(.{
        .name = "test_generated_lib",
        .root_source_file = b.path("tests/Onnx/onnx_loader.zig"),

        .target = target,
        .optimize = optimize,
    });

    test_onnx_parser.root_module.addImport("zant", zant_mod);
    test_onnx_parser.linkLibC();

    const run_test_onnx_parser = b.addRunArtifact(test_onnx_parser);
    const step_test_onnx_parser = b.step("onnx-parser", "Run generated library tests");
    step_test_onnx_parser.dependOn(&run_test_onnx_parser.step);

    // Path to the generated model options file (moved here)
    const model_options_path = std.fmt.allocPrint(b.allocator, "{s}model_options.zig", .{generated_path_option}) catch |err| {
        std.debug.print("Error allocating model options path: {}\n", .{err});
        return;
    };

    // ************************************************ MAIN EXECUTABLE (for profiling) ************************************************

    const main_executable = b.addExecutable(.{
        .name = "main_profiling_target",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    main_executable.linkLibC();
    main_executable.linkLibrary(static_lib);
    const model_opts_mod = b.createModule(.{
        .root_source_file = b.path(model_options_path),
    });
    model_opts_mod.addImport("zant", zant_mod);
    model_opts_mod.addImport("codegen", codeGen_mod);
    main_executable.root_module.addImport("model_opts", model_opts_mod);
    const install_main_exe_step = b.addInstallArtifact(main_executable, .{}); // Installa l'eseguibile

    const build_main_step = b.step("build-main", "Build the main executable for profiling");
    build_main_step.dependOn(&install_main_exe_step.step);

    // ************************************************ NATIVE GUI ************************************************

    {
        const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .sdl, .sdl3 = true });

        const gui_exe = b.addExecutable(.{
            .name = "gui",
            .root_source_file = b.path("gui/sdl/sdl-standalone.zig"),
            .target = target,
            .optimize = optimize,
        });

        // Can either link the backend ourselves:
        // const dvui_mod = dvui_dep.module("dvui");
        // const sdl = dvui_dep.module("sdl");
        // @import("dvui").linkBackend(dvui_mod, sdl);
        // exe.root_module.addImport("dvui", dvui_mod);

        // Or use a prelinked one:
        gui_exe.root_module.addImport("dvui", dvui_dep.module("dvui_sdl"));

        const compile_step = b.step("compile-gui", "Compile gui");
        compile_step.dependOn(&b.addInstallArtifact(gui_exe, .{}).step);
        b.getInstallStep().dependOn(compile_step);

        const run_cmd = b.addRunArtifact(gui_exe);
        run_cmd.step.dependOn(compile_step);

        const run_step = b.step("gui", "Run gui");
        run_step.dependOn(&run_cmd.step);
    }
}
