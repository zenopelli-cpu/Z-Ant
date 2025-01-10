``` zig
    var model = Model(f64){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    defer {
        std.debug.print("\n ------------------------------- model.deinit(); ", .{});
        model.deinit();
    }

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

    const file_path = "./datasets/ZantConvModel1.bin";

    try model_import_export.exportModel(f64, model, file_path);

```