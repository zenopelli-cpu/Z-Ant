const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const Any = zant.uops.Any;

const MemoryRender = @import("memory_render.zig");
const ArithmeticRender = @import("arithmetic_render.zig");
const ReduceRender = @import("reduce_render.zig");
const ConditionalRender = @import("conditional_render.zig");
const UnaryRender = @import("unary_render.zig");
const GepRender = @import("gep_render.zig");
const ControlFlowRender = @import("controlflow_render.zig");
const ViewManager = @import("view_manager.zig");
pub const ViewInfo = ViewManager.ViewInfo;

pub fn ZigRenderer(comptime WriterType: type) type {
    return struct {
        allocator: std.mem.Allocator,
        writer: WriterType,
        view_map: std.AutoHashMap(usize, ViewInfo),

        pub fn init(allocator: std.mem.Allocator, writer: WriterType) @This() {
            return .{
                .allocator = allocator,
                .writer = writer,
                .view_map = std.AutoHashMap(usize, ViewInfo).init(allocator),
            };
        }

        pub fn deinit(self: *@This()) void {
            var iter = self.view_map.iterator();
            while (iter.next()) |entry| {
                const key_ptr = entry.key_ptr;
                _ = self.view_map.removeByPtr(key_ptr);
            }
            self.view_map.deinit();
        }

        pub fn render(self: *@This(), uops: []UOp) !void {
            for (uops) |uop| {
                switch (uop.op) {
                    .DEFINE_GLOBAL, .LOAD, .STORE => try MemoryRender.render(self.allocator, self.writer, uop),
                    .ADD, .SUB, .MUL, .FDIV, .POW => try ArithmeticRender.render(self.allocator, self.writer, uop),
                    .REDUCE_ADD, .REDUCE_MAX => try ReduceRender.render(self.allocator, self.writer, uop),
                    .MAX, .MIN => try ConditionalRender.render(self.allocator, self.writer, uop),
                    .EXP2, .NEG => try UnaryRender.render(self.allocator, self.writer, uop),
                    .GEP => try GepRender.render(self.allocator, self.writer, uop, self.view_map),
                    .RANGE, .ENDRANGE => try ControlFlowRender.render(self.allocator, self.writer, uop),
                    .VIEW => ViewManager.manage(uop, &self.view_map) catch |err| {
                        std.log.debug("Error managing view: {any}\n", .{err});
                        return error.InvalidOperation;
                    },
                    else => {
                        std.log.debug("unknown op {d}\n", .{uop.id});
                        return error.InvalidOperation;
                    },
                }
            }
        }
    };
}
