const std = @import("std");
const Tensor = @import("Tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;

pub fn ConvolutionalLayer(comptime T: type) type {
    return struct {
        // Convolutional layer parameters
        weights: Tensor.Tensor(T), // Weights (kernels) of shape [kernel_shape]
        bias: Tensor.Tensor(T), // Biases for each output channel
        input: Tensor.Tensor(T), // Input tensor
        output: Tensor.Tensor(T), // Output tensor after convolution
        // Layer configuration
        input_channels: usize,
        kernel_shape: [4]usize, // shape : [number of kernel filters, number of channels, height, width ]
        stride: [2]usize,
        // Gradients
        w_gradients: Tensor.Tensor(T),
        b_gradients: Tensor.Tensor(T),
        // Utils
        allocator: *const std.mem.Allocator,

        const Self = @This();

        pub fn create(self: *Self) Layer.Layer(T) {
            return Layer.Layer(T){
                .layer_type = Layer.LayerType.ConvolutionalLayer,
                .layer_ptr = self,
                .layer_impl = &.{
                    .init = init,
                    .deinit = deinit,
                    .forward = forward,
                    .backward = backward,
                    .printLayer = printLayer,
                    .get_n_inputs = get_n_inputs,
                    .get_n_neurons = get_n_neurons,
                    .get_input = get_input,
                    .get_output = get_output,
                },
            };
        }

        /// Initialize the convolutional layer with random weights and biases
        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const argsStruct: *const struct {
                input_channels: usize,
                kernel_shape: [4]usize,
                stride: [2]usize,
            } = @ptrCast(@alignCast(args));

            // Initialize layer configuration
            self.input_channels = argsStruct.input_channels;
            self.kernel_shape = argsStruct.kernel_shape;
            self.allocator = alloc;
            self.stride = argsStruct.stride;

            //assuming kernel is always of size 4
            if (self.kernel_shape.len != 4) {
                std.debug.print("\nERROR : kernel must always be of size 4, while yours is {}, kernel_shape:{any}\nThe values have the following meaning:", .{ self.kernel_shape.len, self.kernel_shape });
                std.debug.print("\n     kernel_shape[0] = number of kernel filters", .{});
                std.debug.print("\n     kernel_shape[1] = number of channels for each filter", .{});
                std.debug.print("\n     kernel_shape[2] = height of each channel", .{});
                std.debug.print("\n     kernel_shape[3] = width of each channel", .{});
                return LayerError.InvalidParameters;
            }

            // Check parameters
            if (self.input_channels <= 0) return LayerError.InvalidParameters;
            if (self.kernel_shape[0] <= 0 or self.kernel_shape[1] <= 0 or self.kernel_shape[2] <= 0 or self.kernel_shape[3] <= 0) return LayerError.InvalidParameters;

            //check channels
            if (self.input_channels != self.kernel_shape[1]) {
                std.debug.print("\nERROR : Input and K ernel must have the same number of channels", .{});
            }

            // Initialize weights and biases
            var bias_shape: [1]usize = [_]usize{self.kernel_shape[0]}; //one bias for each filter

            const weight_array = try Layer.randn(T, self.allocator, 1, self.kernel_shape[0] * self.kernel_shape[1] * self.kernel_shape[2] * self.kernel_shape[3]);
            defer self.allocator.free(weight_array);
            const bias_array = try Layer.randn(T, self.allocator, 1, self.kernel_shape[0]);
            defer self.allocator.free(bias_array);

            self.weights = try Tensor.Tensor(T).fromArray(alloc, weight_array, &self.kernel_shape);
            self.bias = try Tensor.Tensor(T).fromArray(alloc, bias_array, &bias_shape);

            // Initialize gradients to zero
            self.w_gradients = try Tensor.Tensor(T).fromShape(self.allocator, &self.kernel_shape);
            self.b_gradients = try Tensor.Tensor(T).fromShape(self.allocator, &bias_shape);
        }

        /// Deallocate the convolutional layer resources
        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Deallocate tensors if allocated
            if (self.weights.data.len > 0) {
                self.weights.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            if (self.bias.data.len > 0) {
                self.bias.deinit();
            }

            if (self.w_gradients.data.len > 0) {
                self.w_gradients.deinit();
            }

            if (self.b_gradients.data.len > 0) {
                self.b_gradients.deinit();
            }

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            //std.debug.print("\nConvolutionalLayer resources deallocated.\n", .{});
        }

        /// Forward pass of the convolutional layer
        pub fn forward(ctx: *anyopaque, input: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Save input for backward pass
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            // Perform convolution operation
            self.output = try TensMath.convolve_tensor_with_bias(T, T, &self.input, &self.weights, &self.bias, &self.stride);
            //self.output.info();

            return self.output;
        }

        /// Backward pass of the convolutional layer
        pub fn backward(ctx: *anyopaque, dValues: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Initialize gradient tensors if not already initialized
            if (self.w_gradients.data.len > 0) {
                self.w_gradients.deinit();
            }
            if (self.b_gradients.data.len > 0) {
                self.b_gradients.deinit();
            }

            _ = dValues;

            // Compute gradients with respect to biases
            // Sum over the spatial dimensions
            // self.b_gradients = TensMath.convolution_backward_biases(T, dValues) catch |err| {
            //     std.debug.print("Error during conv backward_biases {any}", .{err});
            //     return err;
            // };

            // // Compute gradients with respect to weights
            // self.w_gradients = TensMath.convolution_backward_weights(T, &self.input, dValues) catch |err| {
            //     std.debug.print("Error during conv backward_weights {any}", .{err});
            //     return err;
            // };

            // // Compute gradients with respect to input
            // var dInput = TensMath.convolution_backward_input(T, dValues, &self.weights) catch |err| {
            //     std.debug.print("Error during conv backward_input {any}", .{err});
            //     return err;
            // };
            // _ = &dInput;

            return self.w_gradients; //TODO: CHANGEEEEE!!! JUST FOR DEBUG
        }

        /// Print the convolutional layer information (To be written)
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            _ = ctx;
            _ = choice;
        }

        //---------------------------------------------------------------
        //---------------------------- Getters --------------------------
        //---------------------------------------------------------------
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // For convolutional layers, n_inputs can be considered as input_channels
            return self.input_channels;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // For convolutional layers, n_neurons can be considered as output_channels
            return self.kernel_shape[1];
        }

        pub fn get_weights(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.weights;
        }

        pub fn get_bias(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.bias;
        }

        pub fn get_input(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.output;
        }

        pub fn get_weightGradients(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.w_gradients;
        }

        pub fn get_biasGradients(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            return &self.b_gradients;
        }
    };
}
