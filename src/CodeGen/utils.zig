const std = @import("std");
const zant = @import("zant");
const UOpBuilder = zant.uops.UOpBuilder;
const allocator = zant.utils.allocator.allocator;
const NodeZant = zant.IR_graph.NodeZant;
const operators = zant.IR_graph.operators;
const math = zant.core.tensor.math_standard;
