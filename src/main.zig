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
const InfoAllocator = @import("info_allocator");
const model_import_export = @import("model_import_export");

pub fn main() !void {
    const allocator = @import("pkgAllocator").allocator;

    const file_path = "./datasets/models/model1/ZantConvModel1.zant";

    var model = try model_import_export.importModel(f64, &allocator, file_path);
    defer {
        std.debug.print("\n ------------------------------- imported_model.deinit(); ", .{});
        model.deinit();
    }

    var load = loader.DataLoader(f64, u8, u8, 16, 3){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const image_file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";

    try load.loadMNIST2DDataParallel(&allocator, image_file_name, label_file_name);

    try Trainer.TrainDataLoader2D(
        f64, //The data type for the tensor elements in the model
        u8, //The data type for the input tensor (X)
        u8, //The data type for the output tensor (Y)
        &allocator, //Memory allocator for dynamic allocations during training
        16, //The number of samples in each batch
        784, //The number of features in each input sample
        &model, //A pointer to the model to be trained
        &load, //A pointer to the `DataLoader` that provides data batches
        3, //The total number of epochs to train for
        LossType.CCE, //The type of loss function used during training
        0.005,
        0.8, //Training size
    );

    try model_import_export.exportModel(f64, model, file_path);
}
