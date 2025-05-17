const testing = std.testing;
const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const allocator = zant.utils.allocator.allocator;
const fs = std.fs;

const Tensor = zant.core.tensor.Tensor;

const ErrorDetail = struct {
    modelName: []const u8,
    errorLoad: anyerror,
};

var models: std.ArrayList([]const u8) = std.ArrayList([]const u8).init(allocator);
var failed_parsed_models: std.ArrayList(ErrorDetail) = std.ArrayList(ErrorDetail).init(allocator);
var failed_write_op_models: std.ArrayList(ErrorDetail) = std.ArrayList(ErrorDetail).init(allocator);

test "Test write_op on all oneOp models" {
    std.debug.print("\n     test: Test write_op on all oneOp models\n", .{});

    // Create a temporary file for write_op output
    const temp_file_path = "temp_write_op_test_all.zig";
    var file = try std.fs.cwd().createFile(temp_file_path, .{});
    defer {
        file.close();
        // Delete the temporary file after test
        std.fs.cwd().deleteFile(temp_file_path) catch |err| {
            std.debug.print("Failed to delete temporary file: {}\n", .{err});
        };
    }

    var writer = file.writer();
    try writer.writeAll("// Generated test file for write_op\n");

    // Open oneOp models directory
    var dir = try std.fs.cwd().openDir("generated/oneOpModels", .{ .iterate = true });
    defer dir.close();

    // Iterate over directory entries
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind == .directory) {
            // Add model name to list
            try models.append(try allocator.dupe(u8, entry.name));
        }
    }

    std.debug.print("\n -- Testing write_op on these models: ", .{});
    for (models.items) |model_name| {
        std.debug.print("\n  --- {s}", .{model_name});

        // Format model path according to model_name
        const model_path = try std.mem.concat(allocator, u8, &[_][]const u8{ "generated/oneOpModels/", model_name, "/", model_name, ".onnx" });
        defer allocator.free(model_path);

        try writer.print("\n\n// ---------- Model: {s} ----------\n", .{model_name});

        // Parse the model
        var model = onnx.parseFromFile(allocator, model_path) catch |err| {
            std.debug.print("\nError parsing model {s}: {any}\n", .{ model_name, err });
            try failed_parsed_models.append(.{ .modelName = try allocator.dupe(u8, model_name), .errorLoad = err });
            continue;
        };
        defer model.deinit(allocator);

        // Create GraphZant
        var graphZant = zant.IR_graph.init(&model) catch |err| {
            std.debug.print("\nError creating graph for model {s}: {any}\n", .{ model_name, err });
            try failed_parsed_models.append(.{ .modelName = try allocator.dupe(u8, model_name), .errorLoad = err });
            continue;
        };
        defer graphZant.deinit();

        // Linearize the graph
        const linearizedGraph = graphZant.linearize(allocator) catch |err| {
            std.debug.print("\nError linearizing graph for model {s}: {any}\n", .{ model_name, err });
            try failed_parsed_models.append(.{ .modelName = try allocator.dupe(u8, model_name), .errorLoad = err });
            continue;
        };
        defer linearizedGraph.deinit();

        std.debug.print("\n  Linearized Graph for {s} has {d} nodes\n", .{ model_name, linearizedGraph.items.len });

        // Test write_op on each node
        for (linearizedGraph.items, 0..) |node, i| {
            const node_name = node.name orelse "<unnamed>";
            std.debug.print("\n   - Testing node {d}: {s} (type: {s})", .{ i, node_name, node.op_type });

            try writer.print("\n// Node {d}: {s} (type: {s})\n", .{ i, node_name, node.op_type });

            // Try to call write_op on this node
            node.write_op(writer) catch |err| {
                std.debug.print("\n     ERROR: Failed to write_op for node {s} in model {s}: {any}\n", .{ node_name, model_name, err });
                try failed_write_op_models.append(.{ .modelName = try std.fmt.allocPrint(allocator, "{s}::{s}", .{ model_name, node_name }), .errorLoad = err });
                try writer.print("// ERROR: write_op failed with error: {any}\n", .{err});
                continue;
            };

            try writer.print("// write_op completed successfully\n", .{});
        }
    }

    if (failed_parsed_models.items.len != 0) {
        std.debug.print("\n\n FAILED ONNX PARSED MODELS: ", .{});
        for (failed_parsed_models.items) |fm|
            std.debug.print("\n model:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
    }

    if (failed_write_op_models.items.len != 0) {
        std.debug.print("\n\n FAILED WRITE_OP OPERATIONS: ", .{});
        for (failed_write_op_models.items) |fm|
            std.debug.print("\n model/node:{s} error:{any}", .{ fm.modelName, fm.errorLoad });
    }

    if (failed_parsed_models.items.len == 0 and failed_write_op_models.items.len == 0) {
        std.debug.print("\n\n ---- SUCCESSFULLY TESTED WRITE_OP ON ALL ONE-OP MODELS ---- \n\n", .{});
    }

    // Clean up
    for (models.items) |name| {
        allocator.free(name);
    }
    models.deinit();

    for (failed_parsed_models.items) |item| {
        allocator.free(item.modelName);
    }
    failed_parsed_models.deinit();

    for (failed_write_op_models.items) |item| {
        allocator.free(item.modelName);
    }
    failed_write_op_models.deinit();
}
