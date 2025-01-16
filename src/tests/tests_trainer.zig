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
const expect = std.testing.expect;

// test "Test single epoch training with simplified CNN" {
//     var allocator = std.testing.allocator;

//     var model = Model(f64){
//         .layers = undefined,
//         .allocator = &allocator,
//         .input_tensor = undefined,
//     };
//     try model.init();
//     defer model.deinit();

//     // Conv layer 1
//     var conv_layer = convlayer(f64){
//         .weights = undefined,
//         .bias = undefined,
//         .input = undefined,
//         .output = undefined,
//         .input_channels = 0,
//         .kernel_shape = undefined,
//         .stride = undefined,
//         .w_gradients = undefined,
//         .b_gradients = undefined,
//         .allocator = &allocator,
//     };
//     var layer_ = conv_layer.create();
//     try layer_.init(
//         &allocator,
//         @constCast(&struct {
//             input_channels: usize,
//             kernel_shape: [4]usize,
//             stride: [2]usize,
//         }{
//             .input_channels = 1,
//             .kernel_shape = .{ 8, 1, 2, 2 },
//             .stride = .{ 1, 1 },
//         }),
//     );
//     try model.addLayer(layer_);

//     // Pooling layer
//     var pool = PoolingLayer(f64){
//         .input = undefined,
//         .output = undefined,
//         .used_input = undefined,
//         .kernel = .{ 2, 2 },
//         .stride = .{ 2, 2 },
//         .poolingType = .Max,
//         .allocator = &allocator,
//     };
//     var pool_layer = try pool.create();
//     try pool_layer.init(&allocator, @constCast(&struct {
//         kernel: [2]usize,
//         stride: [2]usize,
//         poolingType: PoolingType,
//     }{
//         .kernel = .{ 2, 2 },
//         .stride = .{ 2, 2 },
//         .poolingType = .Max,
//     }));
//     try model.addLayer(pool_layer);

//     // Flatten layer
//     var flatten = flattenlayer(f64){
//         .input = undefined,
//         .output = undefined,
//         .allocator = &allocator,
//         .original_shape = &[_]usize{},
//     };
//     var flat_layer = flatten.create();
//     try flat_layer.init(&allocator, @constCast(&struct {
//         placeholder: bool,
//     }{
//         .placeholder = true,
//     }));
//     try model.addLayer(flat_layer);

//     // Dense layer
//     var dense = denselayer(f64){
//         .weights = undefined,
//         .bias = undefined,
//         .input = undefined,
//         .output = undefined,
//         .n_inputs = 0,
//         .n_neurons = 0,
//         .w_gradients = undefined,
//         .b_gradients = undefined,
//         .allocator = undefined,
//     };
//     var dense_ = denselayer(f64).create(&dense);

//     // Input size calculation:
//     // 1. Conv (8 filters 2x2, stride 1): 27x27x8
//     // 2. MaxPool (2x2, stride 2): 13x13x8
//     // 3. Flatten: 13 * 13 * 8 = 1352
//     try dense_.init(&allocator, @constCast(&struct {
//         n_inputs: usize,
//         n_neurons: usize,
//     }{
//         .n_inputs = 13 * 13 * 8, // Corretto calcolo delle dimensioni
//         .n_neurons = 10,
//     }));
//     try model.addLayer(dense_);

//     // Setup DataLoader with smaller batch size for testing
//     var load = loader.DataLoader(f64, u8, u8, 16, 3){
//         .X = undefined,
//         .y = undefined,
//         .xTensor = undefined,
//         .yTensor = undefined,
//         .XBatch = undefined,
//         .yBatch = undefined,
//     };

//     const image_file = "datasets/t10k-images-idx3-ubyte";
//     const label_file = "datasets/t10k-labels-idx1-ubyte";
//     try load.loadMNIST2DDataParallel(&allocator, image_file, label_file);
//     defer load.deinit(&allocator);

//     // Train for 1 epoch
//     try Trainer.TrainDataLoader2D(
//         f64,
//         u8,
//         u8,
//         &allocator,
//         16,
//         784,
//         &model,
//         &load,
//         1,
//         LossType.CCE,
//         0.005,
//         0.9,
//     );
// }
