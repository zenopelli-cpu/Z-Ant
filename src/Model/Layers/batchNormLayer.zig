const std = @import("std");
const Tensor = @import("Tensor");
const TensorMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;

/// Represents a batch normalization layer in a neural network.
/// This layer normalizes the activations of the previous layer at each batch,
/// making the network more stable by reducing internal covariate shift.
///
/// @param T The data type of the tensor elements (e.g., `f32`, `f64`, etc.).
pub fn BatchNormLayer(comptime T: type) type {
    return struct {
        // Layer parameters
        gamma: Tensor.Tensor(T), // Scale parameter
        beta: Tensor.Tensor(T), // Shift parameter
        input: Tensor.Tensor(T), // Input tensor
        output: Tensor.Tensor(T), // Output tensor
        running_mean: Tensor.Tensor(T), // Running mean for inference
        running_var: Tensor.Tensor(T), // Running variance for inference
        epsilon: T, // Small constant for numerical stability
        momentum: T, // Momentum for running statistics
        is_training: bool, // Whether the layer is in training mode
        allocator: *const std.mem.Allocator,

        // Gradients
        gamma_grad: Tensor.Tensor(T),
        beta_grad: Tensor.Tensor(T),

        // Cache for backward pass
        normalized: Tensor.Tensor(T),
        std_dev: Tensor.Tensor(T),
        var_: Tensor.Tensor(T),
        mean: Tensor.Tensor(T),

        const Self = @This();

        pub const BatchNormInitArgs = struct {
            num_features: usize,
            epsilon: T = 1e-5,
            momentum: T = 0.1,
        };

        pub fn create(self: *Self) Layer.Layer(T) {
            return Layer.Layer(T){
                .layer_type = Layer.LayerType.BatchNormLayer,
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

        pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const argsStruct: *const BatchNormInitArgs = @ptrCast(@alignCast(args));

            self.allocator = alloc;
            self.epsilon = argsStruct.epsilon;
            self.momentum = argsStruct.momentum;
            self.is_training = true;

            // Initialize gamma and beta to match the feature dimension
            var shape = [_]usize{argsStruct.num_features}; // Keep as 1D tensor

            // Initialize gamma to ones
            const gamma_data = try self.allocator.alloc(T, argsStruct.num_features);
            @memset(gamma_data, 1);
            self.gamma = try Tensor.Tensor(T).fromArray(self.allocator, gamma_data, &shape);
            self.allocator.free(gamma_data);

            // Initialize beta to zeros
            const beta_data = try self.allocator.alloc(T, argsStruct.num_features);
            @memset(beta_data, 0);
            self.beta = try Tensor.Tensor(T).fromArray(self.allocator, beta_data, &shape);
            self.allocator.free(beta_data);

            // Initialize running statistics
            const zeros = try self.allocator.alloc(T, argsStruct.num_features);
            @memset(zeros, 0);
            self.running_mean = try Tensor.Tensor(T).fromArray(self.allocator, zeros, &shape);
            self.running_var = try Tensor.Tensor(T).fromArray(self.allocator, zeros, &shape);
            self.allocator.free(zeros);

            // Initialize gradients
            self.gamma_grad = try Tensor.Tensor(T).fromShape(self.allocator, &shape);
            self.beta_grad = try Tensor.Tensor(T).fromShape(self.allocator, &shape);

            // Initialize cache tensors
            self.normalized = try Tensor.Tensor(T).fromShape(self.allocator, &shape);
            self.std_dev = try Tensor.Tensor(T).fromShape(self.allocator, &shape);
            self.var_ = try Tensor.Tensor(T).fromShape(self.allocator, &shape);
            self.mean = try Tensor.Tensor(T).fromShape(self.allocator, &shape);
        }

        pub fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.gamma.data.len > 0) self.gamma.deinit();
            if (self.beta.data.len > 0) self.beta.deinit();
            if (self.input.data.len > 0) self.input.deinit();
            if (self.output.data.len > 0) self.output.deinit();
            if (self.running_mean.data.len > 0) self.running_mean.deinit();
            if (self.running_var.data.len > 0) self.running_var.deinit();
            if (self.gamma_grad.data.len > 0) self.gamma_grad.deinit();
            if (self.beta_grad.data.len > 0) self.beta_grad.deinit();
            if (self.normalized.data.len > 0) self.normalized.deinit();
            if (self.std_dev.data.len > 0) self.std_dev.deinit();
            if (self.var_.data.len > 0) self.var_.deinit();
            if (self.mean.data.len > 0) self.mean.deinit();
        }

        pub fn forward(ctx: *anyopaque, input: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Store input for backward pass
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            // Calculate mean for each feature
            var mean_data = try self.allocator.alloc(T, input.shape[1]);
            defer self.allocator.free(mean_data);
            @memset(mean_data, 0);

            const batch_size = @as(T, @floatFromInt(input.shape[0]));
            for (0..input.shape[1]) |feature| {
                var sum: T = 0;
                for (0..input.shape[0]) |batch| {
                    sum += input.data[batch * input.shape[1] + feature];
                }
                mean_data[feature] = sum / batch_size;
            }

            // Calculate variance for each feature
            var var_data = try self.allocator.alloc(T, input.shape[1]);
            defer self.allocator.free(var_data);
            @memset(var_data, 0);

            for (0..input.shape[1]) |feature| {
                var sum_sq: T = 0;
                const mean = mean_data[feature];
                for (0..input.shape[0]) |batch| {
                    const diff = input.data[batch * input.shape[1] + feature] - mean;
                    sum_sq += diff * diff;
                }
                var_data[feature] = sum_sq / batch_size;
            }

            // Normalize input and store for backward pass
            if (self.normalized.data.len > 0) {
                self.normalized.deinit();
            }
            self.normalized = try input.copy();
            for (0..input.shape[0]) |batch| {
                for (0..input.shape[1]) |feature| {
                    const idx = batch * input.shape[1] + feature;
                    self.normalized.data[idx] = (input.data[idx] - mean_data[feature]) /
                        @sqrt(var_data[feature] + self.epsilon);
                }
            }

            // Store mean and var for backward pass
            if (self.mean.data.len > 0) {
                self.mean.deinit();
            }
            var mean_shape = [_]usize{ 1, input.shape[1] };
            self.mean = try Tensor.Tensor(T).fromArray(self.allocator, mean_data, &mean_shape);

            if (self.var_.data.len > 0) {
                self.var_.deinit();
            }
            var var_shape = [_]usize{ 1, input.shape[1] };
            self.var_ = try Tensor.Tensor(T).fromArray(self.allocator, var_data, &var_shape);

            // Scale and shift using gamma and beta
            var output_data = try self.allocator.alloc(T, input.data.len);
            defer self.allocator.free(output_data);

            for (0..input.shape[0]) |batch| {
                for (0..input.shape[1]) |feature| {
                    const idx = batch * input.shape[1] + feature;
                    output_data[idx] = self.normalized.data[idx] * self.gamma.data[feature] +
                        self.beta.data[feature];
                }
            }

            // Update running statistics during training
            if (self.is_training) {
                for (0..input.shape[1]) |feature| {
                    self.running_mean.data[feature] = (1 - self.momentum) * self.running_mean.data[feature] +
                        self.momentum * mean_data[feature];
                    self.running_var.data[feature] = (1 - self.momentum) * self.running_var.data[feature] +
                        self.momentum * var_data[feature];
                }
            }

            // Create and return output tensor
            if (self.output.data.len > 0) {
                self.output.deinit();
            }
            self.output = try Tensor.Tensor(T).fromArray(self.allocator, output_data, input.shape);
            return self.output;
        }

        pub fn backward(ctx: *anyopaque, dvalues: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Gradients with respect to gamma and beta
            var dgamma_data = try self.allocator.alloc(T, self.input.shape[1]);
            defer self.allocator.free(dgamma_data);
            var dbeta_data = try self.allocator.alloc(T, self.input.shape[1]);
            defer self.allocator.free(dbeta_data);
            @memset(dgamma_data, 0);
            @memset(dbeta_data, 0);

            for (0..self.input.shape[1]) |feature| {
                for (0..self.input.shape[0]) |batch| {
                    const idx = batch * self.input.shape[1] + feature;
                    dgamma_data[feature] += dvalues.data[idx] * self.normalized.data[idx];
                    dbeta_data[feature] += dvalues.data[idx];
                }
            }

            // Store gradients
            if (self.gamma_grad.data.len > 0) {
                self.gamma_grad.deinit();
            }
            var dgamma_shape = [_]usize{self.input.shape[1]};
            self.gamma_grad = try Tensor.Tensor(T).fromArray(self.allocator, dgamma_data, &dgamma_shape);

            if (self.beta_grad.data.len > 0) {
                self.beta_grad.deinit();
            }
            var dbeta_shape = [_]usize{self.input.shape[1]};
            self.beta_grad = try Tensor.Tensor(T).fromArray(self.allocator, dbeta_data, &dbeta_shape);

            // Calculate gradients with respect to input
            var dx_data = try self.allocator.alloc(T, self.input.data.len);
            defer self.allocator.free(dx_data);

            for (0..self.input.shape[0]) |batch| {
                for (0..self.input.shape[1]) |feature| {
                    const idx = batch * self.input.shape[1] + feature;
                    const std_dev = @sqrt(self.var_.data[feature] + self.epsilon);
                    dx_data[idx] = dvalues.data[idx] * self.gamma.data[feature] / std_dev;
                }
            }

            return Tensor.Tensor(T).fromArray(self.allocator, dx_data, self.input.shape);
        }

        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            switch (choice) {
                0 => std.debug.print("BatchNorm Layer\n", .{}),
                1 => {
                    std.debug.print("gamma: ", .{});
                    self.gamma.printMultidim();
                    std.debug.print("\nbeta: ", .{});
                    self.beta.printMultidim();
                    std.debug.print("\nrunning_mean: ", .{});
                    self.running_mean.printMultidim();
                    std.debug.print("\nrunning_var: ", .{});
                    self.running_var.printMultidim();
                },
                else => {},
            }
        }

        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return if (self.input.shape.len > 1) self.input.shape[1] else 0;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            return get_n_inputs(ctx);
        }

        pub fn get_input(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *Tensor.Tensor(T) {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return &self.output;
        }

        pub fn set_training(ctx: *anyopaque, is_training: bool) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.is_training = is_training;
        }
    };
}
