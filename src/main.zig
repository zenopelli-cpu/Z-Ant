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

pub fn main() !void {
    const allocator = @import("pkgAllocator").allocator;

    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    //layer 0: First Convolutional Layer ----------------------------------------------------------------------------
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
            .kernel_shape = .{ 16, 1, 2, 2 }, // 32 filters, 1 channel, 3x3 kernel
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer_);

    //layer 1: ReLU Activation ----------------------------------------------------------------------------
    // var conv1_activ = activationlayer(f64){
    //     .input = undefined,
    //     .output = undefined,
    //     .n_inputs = 0,
    //     .n_neurons = 0,
    //     .activationFunction = ActivationType.ReLU,
    //     .allocator = &allocator,
    // };
    // var conv1_act = activationlayer(f64).create(&conv1_activ);
    // try conv1_act.init(&allocator, @constCast(&struct {
    //     n_inputs: usize,
    //     n_neurons: usize,
    // }{
    //     .n_inputs = 16,
    //     .n_neurons = 16,
    // }));
    // try model.addLayer(conv1_act);

    //layer 2: Second Convolutional Layer ----------------------------------------------------------------------------
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
            .kernel_shape = .{ 16, 16, 2, 2 }, // 64 filters, 32 channels, 3x3 kernel
            .stride = .{ 1, 1 },
        }),
    );
    try model.addLayer(layer2_);

    //layer 3: ReLU Activation ----------------------------------------------------------------------------
    // var conv2_activ = activationlayer(f64){
    //     .input = undefined,
    //     .output = undefined,
    //     .n_inputs = 0,
    //     .n_neurons = 0,
    //     .activationFunction = ActivationType.ReLU,
    //     .allocator = &allocator,
    // };
    // var conv2_act = activationlayer(f64).create(&conv2_activ);
    // try conv2_act.init(&allocator, @constCast(&struct {
    //     n_inputs: usize,
    //     n_neurons: usize,
    // }{
    //     .n_inputs = 64,
    //     .n_neurons = 64,
    // }));
    // try model.addLayer(conv2_act);

    //layer 5: Flatten Layer ----------------------------------------------------------------------------
    var flatten_layer = flattenlayer(f64){
        .input = undefined,
        .output = undefined,
        .allocator = &allocator,
        .original_shape = &[_]usize{},
    };
    var Flattenlayer = flatten_layer.create();

    // Initialize the Flatten layer with placeholder args
    var init_args = flattenlayer(f64).FlattenInitArgs{
        .placeholder = true,
    };
    try Flattenlayer.init(&allocator, &init_args);
    try model.addLayer(Flattenlayer);

    //layer 6: Dense Layer (512 neurons) ----------------------------------------------------------------------------
    var dense1 = denselayer(f64){
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
    var dense1_ = denselayer(f64).create(&dense1);
    try dense1_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10816, // Calculated from previous layer output
        .n_neurons = 256,
    }));
    try model.addLayer(dense1_);

    //layer 7: ReLU Activation ----------------------------------------------------------------------------
    var dense1_activ = activationlayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 10816,
        .n_neurons = 256,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var dense1_act = activationlayer(f64).create(&dense1_activ);
    try dense1_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 10816,
        .n_neurons = 256,
    }));
    try model.addLayer(dense1_act);

    //layer 8: Output Dense Layer (10 neurons for MNIST) ----------------------------------------------------------------------------
    var dense2 = denselayer(f64){
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
    var dense2_ = denselayer(f64).create(&dense2);
    try dense2_.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 256,
        .n_neurons = 10,
    }));
    try model.addLayer(dense2_);

    //layer 9: Softmax Activation ----------------------------------------------------------------------------
    var output_activ = activationlayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 256,
        .n_neurons = 10,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var output_act = activationlayer(f64).create(&output_activ);
    try output_act.init(&allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 256,
        .n_neurons = 10,
    }));
    try model.addLayer(output_act);

    // Setup DataLoader
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

    // Train the model
    try Trainer.TrainDataLoader2D(
        f64,
        u8,
        u8,
        &allocator,
        16,
        784,
        &model,
        &load,
        10,
        LossType.CCE,
        0.005,
        0.8,
    );

    model.deinit();
}
