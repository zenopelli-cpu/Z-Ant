const std = @import("std");
const tensor = @import("tensor");
//--- layers
const layer = @import("layer");
const DenseLayer = layer.DenseLayer;
const ConvolutionalLayer = layer.ConvolutionalLayer;
const FlattenLayer = layer.FlattenLayer;
const PoolingLayer = layer.PoolingLayer;
const PoolingType = layer.poolingLayer.PoolingType;
const ActivationLayer = layer.ActivationLayer;
const BatchNormLayer = layer.BatchNormLayer;
//--- other
const Model = @import("model").Model;
const loader = @import("dataloader");
const ActivationType = @import("activation_function").ActivationType;
const LossType = @import("loss").LossType;
const Trainer = @import("trainer");

pub fn main() !void {
    const allocator = @import("pkgAllocator").allocator;

    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    //layer 0: First Convolutional Layer ----------------------------------------------------------------------------
    var conv_layer = ConvolutionalLayer(f64){
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
            .kernel_shape = .{ 16, 1, 5, 5 }, // 16 filters, 1 channel, 5x5 kernel
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer_);

    // After first conv layer
    var conv1_activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 16 * 24 * 24, // Output size from conv1
        .n_neurons = 16 * 24 * 24,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var conv1_act = ActivationLayer(f64).create(&conv1_activ);
    try conv1_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 16 * 24 * 24,
        .n_neurons = 16 * 24 * 24,
    }));
    try model.addLayer(conv1_act);

    //layer 1: Second Convolutional Layer ----------------------------------------------------------------------------
    var conv_layer2 = ConvolutionalLayer(f64){
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
            .kernel_shape = .{ 32, 16, 5, 5 }, // 32 filters, 16 channels, 5x5 kernel
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer2_);

    // After second conv layer
    var conv2_activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 32 * 20 * 20, // Output size from conv2
        .n_neurons = 32 * 20 * 20,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var conv2_act = ActivationLayer(f64).create(&conv2_activ);
    try conv2_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 32 * 20 * 20,
        .n_neurons = 32 * 20 * 20,
    }));
    try model.addLayer(conv2_act);

    // MaxPool after convs
    var pool1 = PoolingLayer(f64){
        .input = undefined,
        .output = undefined,
        .used_input = undefined,
        .kernel = .{ 2, 2 },
        .stride = .{ 2, 2 },
        .poolingType = .Max,
        .allocator = &allocator,
    };
    var pool1_layer = try pool1.create();

    const PoolInitArgs = struct {
        kernel: [2]usize,
        stride: [2]usize,
        poolingType: PoolingType,
    };
    var pool1_init_args = PoolInitArgs{
        .kernel = .{ 2, 2 },
        .stride = .{ 2, 2 },
        .poolingType = .Max,
    };
    try pool1_layer.init(&allocator, @constCast(&pool1_init_args));
    try model.addLayer(pool1_layer);

    //layer 5: Flatten Layer ----------------------------------------------------------------------------
    var flatten_layer = FlattenLayer(f64){
        .input = undefined,
        .output = undefined,
        .allocator = &allocator,
        .original_shape = &[_]usize{},
    };
    var Flattenlayer = flatten_layer.create();

    const FlattenInitArgs = struct {
        placeholder: bool,
    };
    var flatten_init_args = FlattenInitArgs{
        .placeholder = true,
    };
    try Flattenlayer.init(&allocator, @constCast(&flatten_init_args));
    try model.addLayer(Flattenlayer);

    //layer 6: Dense Layer (128 neurons) ----------------------------------------------------------------------------
    var dense1 = DenseLayer(f64){
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
    var dense1_ = DenseLayer(f64).create(&dense1);
    try dense1_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 3200, // Calculated: 10x10x32 / 4 (after maxpool)
        .n_neurons = 128,
    }));
    try model.addLayer(dense1_);

    //layer 7: ReLU Activation ----------------------------------------------------------------------------
    var dense1_activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 128,
        .n_neurons = 128,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var dense1_act = ActivationLayer(f64).create(&dense1_activ);
    try dense1_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 128,
        .n_neurons = 128,
    }));
    try model.addLayer(dense1_act);

    //layer 8: Output Dense Layer (10 neurons for MNIST) ----------------------------------------------------------------------------
    var dense2 = DenseLayer(f64){
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
    var dense2_ = DenseLayer(f64).create(&dense2);
    try dense2_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 128,
        .n_neurons = 10,
    }));
    try model.addLayer(dense2_);

    //layer 9: Softmax Activation ----------------------------------------------------------------------------
    var output_activ = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 10,
        .n_neurons = 10,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var output_act = ActivationLayer(f64).create(&output_activ);
    try output_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10,
        .n_neurons = 10,
    }));
    try model.addLayer(output_act);

    // Setup DataLoader
    var load = loader.DataLoader(f64, u8, u8, 32, 3){
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

    // Train the model with adjusted hyperparameters
    try Trainer.TrainDataLoader2D(
        f64,
        u8,
        u8,
        &allocator,
        32,
        784,
        &model,
        &load,
        15,
        LossType.CCE,
        0.005,
        0.8,
        0.00001,
        1.0,
    );

    model.deinit();
}
