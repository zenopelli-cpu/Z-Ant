const std = @import("std");

pub const Codegen_flags = struct {
    model_name_option: []const u8,
    model_path_option: []const u8,
    generated_path_option: []const u8,
    user_tests_option: bool,
    log_option: bool,
    input_type_option: []const u8,
    output_type_option: []const u8,
    comm_option: bool,
    dynamic_option: bool,
    //static_planning_option: bool,
    fuse_option: bool,
    export_option: bool,
    codegen_version_option: []const u8,
    xip_enabled: bool,
    use_tensor_pool: bool,

    pub fn init(b: *std.Build) !Codegen_flags {
        const model_name_option = b.option([]const u8, "model", "Model name") orelse "mnist-8";

        var generated_path_option = b.option([]const u8, "generated_path", "Generated path") orelse "";
        if (generated_path_option.len == 0) {
            generated_path_option = std.fmt.allocPrint(b.allocator, "generated/{s}/", .{model_name_option}) catch |err| {
                std.log.scoped(.build).warn("Error allocating generated path: {}\n", .{err});
                return err;
            };
        } else {
            if (!std.mem.endsWith(u8, generated_path_option, "/")) {
                generated_path_option = std.fmt.allocPrint(b.allocator, "{s}/", .{generated_path_option}) catch |err| {
                    std.log.scoped(.build).warn("Error normalizing path: {}\n", .{err});
                    return err;
                };
            }
            generated_path_option = std.fmt.allocPrint(b.allocator, "{s}{s}/", .{ generated_path_option, model_name_option }) catch |err| {
                std.log.scoped(.build).warn("Error allocating generated path: {}\n", .{err});
                return err;
            };
        }
        return Codegen_flags{
            .model_name_option = model_name_option,
            .model_path_option = b.option([]const u8, "model_path", "Model path") orelse std.fmt.allocPrint(b.allocator, "datasets/models/{s}/{s}.onnx", .{ model_name_option, model_name_option }) catch |err| {
                std.log.scoped(.build).warn("Error allocating model path: {}\n", .{err});
                return err;
            },
            // Generated path
            .generated_path_option = generated_path_option,
            .user_tests_option = b.option(bool, "enable_user_tests", "User tests path") orelse false,
            .log_option = b.option(bool, "log", "Run with log") orelse false,
            .input_type_option = b.option([]const u8, "type", "Input type") orelse "f32",
            .output_type_option = b.option([]const u8, "output_type", "Output type") orelse "f32",
            .comm_option = b.option(bool, "comm", "Codegen with comments") orelse false,
            .dynamic_option = b.option(bool, "dynamic", "Dynamic allocation") orelse false,
            //.static_planning_option = b.option(bool, "static_planning", "Perform static memory planning to optimize memory allocation (ignored when -dynamic=true)") orelse true,
            .fuse_option = b.option(bool, "fuse", "enable Kernel fusion") orelse false,
            .export_option = b.option(bool, "do_export", "codegen Exportable ") orelse false,
            .codegen_version_option = b.option([]const u8, "v", "Version, v1 or v2") orelse "v1",
            // XIP (Execute In Place) support for neural network weights
            .xip_enabled = b.option(bool, "xip", "Enable XIP (Execute In Place) for neural network weights") orelse false,
            .use_tensor_pool = b.option(bool, "use_tensor_pool", "Allocate large tensor arrays to tensor_pool section for embedded targets") orelse false,
        };
    }
};
