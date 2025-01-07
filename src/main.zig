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

pub fn main() !void {
    const allocator = @import("pkgAllocator").allocator;

    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    //layer 0 ----------------------------------------------------------------------------
    var conv_layer = convlayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .input_channels = 0,
        .kernel_shape = undefined,
        .stride = undefined,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = &allocator,
    };
    var layer_ = conv_layer.create();
    try layer_.init(
        &allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 1,
            .kernel_shape = .{ 16, 1, 2, 2 }, //filters, channels, rows, cols
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer_);

    //layer 1 ----------------------------------------------------------------------------
    var conv_layer2 = convlayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .input_channels = 0,
        .kernel_shape = undefined,
        .stride = undefined,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = &allocator,
    };
    var layer2_ = conv_layer2.create();
    try layer2_.init(
        &allocator,
        @constCast(&struct {
            input_channels: usize,
            kernel_shape: [4]usize,
            stride: [2]usize,
        }{
            .input_channels = 16,
            .kernel_shape = .{ 16, 16, 3, 3 }, //filters, channels, rows, cols
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer2_);

    //layer 2 ----------------------------------------------------------------------------
    var flatten_layer = flattenlayer(f64){
        .input = undefined,
        .output = undefined,
        .allocator = &allocator,
    };
    var Flattenlayer = flatten_layer.create();

    // Initialize the Flatten layer with placeholder args
    var init_argsF = flattenlayer(f64).FlattenInitArgs{
        .placeholder = true,
    };
    try Flattenlayer.init(&allocator, &init_argsF);

    try model.addLayer(Flattenlayer);

    //layer 3 ----------------------------------------------------------------------------
    var layer3 = denselayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var layer3_ = denselayer(f64).create(&layer3);
    try layer3_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10000,
        .n_neurons = 256,
    }));
    try model.addLayer(layer3_);

    //layer 4 ----------------------------------------------------------------------------
    var layer3Activ = activationlayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer3_act = activationlayer(f64).create(&layer3Activ);
    try layer3_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10000,
        .n_neurons = 256,
    }));
    try model.addLayer(layer3_act);

    //new dense layer

    //layer 5 ----------------------------------------------------------------------------
    var layer4 = denselayer(f64){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };

    var layer4_ = denselayer(f64).create(&layer4);
    try layer4_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 256,
        .n_neurons = 10,
    }));

    try model.addLayer(layer4_);

    //layer 6 ----------------------------------------------------------------------------
    var layer4Activ = activationlayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var layer4_act = activationlayer(f64).create(&layer4Activ);
    try layer4_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 256,
        .n_neurons = 10,
    }));
    try model.addLayer(layer4_act);

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

    model.deinit();
}
