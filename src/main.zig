const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layer");
const denselayer = @import("denselayer").DenseLayer;
const convlayer = @import("convLayer").ConvolutionalLayer;
const flattenlayer = @import("flattenLayer").FlattenLayer;
const PoolingLayer = @import("poolingLayer").PoolingLayer;
const PoolingType = @import("poolingLayer").PoolingType;
const activationlayer = @import("activationlayer").ActivationLayer;
const Model = @import("model").Model;
const loader = @import("dataloader");
const ActivationType = @import("activation_function").ActivationType;
const LossType = @import("loss").LossType;
const Trainer = @import("trainer");
const BatchNormLayer = @import("batchNormLayer").BatchNormLayer;
const onnx = @import("DataHandler/onnx.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var model = try onnx.parseFromFile(allocator, "/home/marco/TheTinyBook/datasets/mnist-8.onnx");
    defer model.deinit(allocator);

    onnx.printStructure(&model);
}
