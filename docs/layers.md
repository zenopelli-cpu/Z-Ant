# Layer

This folder contains the implementation of various neural network layers.
Each file defines a struct representing a specific type of layer however each layers adheres to commn methods such as `init`, `deinit`, `forward` and `backward`

## Layers Type
1. [Convolution Layer](#convlayerzig)
2. [Activation Layer](#activationlayerzig)
3. [Pooling Layer](#poolinglayerzig)
4. [Batch Normalization Layer](#batchnormlayerzig)
5. [Flatten Layer](#flattenlayerzig)
6. [Dense Layer](#denselayerzig)

## convLayer.zig
This module implements a **Convolutional layer**, a fundamental layer type in many deep learning architectures.

### What is the point of convolution?
A convolutional layer in a neural network is used to extract important features from input data, such as images, by applying learnable filters (kernels). These filters detect patterns like edges, textures, and shapes, enabling the network to understand spatial hierarchies and improve object recognition.

### Mathematical explanation
The detailed mathematical explanation of the convolution operation is avaiable [here](#covolution-link)

### In Depth Function Documentation

#### `pub fn ConvLayer(comptime T: type) type`
Initializes and returns a struct that implements convolution operations, adhering to the `Layer` interface.

##### Methods

1. **`pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void`**  
   Allocates the necessary memory for convolution kernels (filters) and biases and randomly initializes them.  
   - **Errors**:  
     - Allocation failures if creating filter tensors or bias tensors fails.

2. **`pub fn deinit(ctx: *anyopaque) void`**  
   Frees memory used by filters, biases, and any intermediate buffers.

3. **`pub fn forward(ctx: *anyopaque, input: *Tensor(T)) !Tensor(T)`**  
   Applies convolution on the input tensor using the stored filters and biases.  
   - **Parameters**:  
     - `input`: An input tensor, typically shaped `[batch_size, channels, height, width]`.  
   - **Returns**:  
     - A new tensor representing the convolved output.  
   - **Errors**:  
     - Shape mismatch (e.g., if input does not match expected channel dimensions), memory errors.

4. **`pub fn backward(ctx: *anyopaque, dValues: *Tensor(T)) !Tensor(T)`**  
   Computes the gradients for the filters and biases, returning the gradient to propagate to the previous layer.  
   - **Parameters**:  
     - `dValues`: Gradient flowing from the subsequent layer, typically shaped like the forward output.  
   - **Returns**:  
     - A gradient tensor matching the shape of the layer’s input.  
   - **Errors**:  
     - Memory and shape mismatches.

5. **`pub fn printLayer(ctx: *anyopaque, choice: u8) void`**  
   Prints relevant convolution parameters such as kernel size, stride, padding, number of filters, etc. for debugging purposes.

6. **Other getters** (`get_n_inputs`, `get_n_neurons`, `get_weights`, `get_bias`, `get_input`, `get_input`, `get_weightGradients`, `get_biasGradients`) 

---

## activationLayer.zig
This module implements various **activation functions** (ReLU, Sigmoid, Tanh, etc.) used to introduce non-linearity into neural networks.

### What is the point of activation?
An activation layer introduces nonlinearity into a neural network, helping it learn complex, non-linear mappings between inputs and outputs. By transforming the summed input signals, activation functions enable the network to capture more sophisticated patterns and relationships that a purely linear model would miss.

<!-- 
### Aviable activation functions
1. **ReLU (Rectified Linear Unit)**  
   \[
   \mathrm{ReLU}(x) = \max(0, x)
   \]

2. **Sigmoid**  
   \[
   \sigma(x) = \frac{1}{1 + e^{-x}}
   \]

3. **Leaky ReLU**  
   \[
   \mathrm{LeakyReLU}(x) =
   \begin{cases}
     x & \text{se } x > 0 \\
     \alpha x & \text{se } x \leq 0
   \end{cases}
   \]
   dove \(\alpha\) è un valore iperparametro (ad esempio 0.01).

4. **Softmax**  
   \[
   \mathrm{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
   \]
   Applicata ad un vettore \( \mathbf{x} \), normalizza le uscite affinché la loro somma sia 1, interpretandole come probabilità.   -->


### In Depth Function Documentation

#### `pub fn ActivationLayer(comptime T: type, activationType: ActivationType) type`
Initializes and returns a struct for a chosen activation function, implementing the `Layer` interface.

##### Methods

1. **`pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void`**  
   Perform minimal initialization.

2. **`pub fn deinit(ctx: *anyopaque) void`**  
   Frees any resources if used (commonly none for standard activations).

3. **`pub fn forward(ctx: *anyopaque, input: *Tensor(T)) !Tensor(T)`**  
   Applies the chosen activation function element-wise to `input`.  
   - **Parameters**:  
     - `input`: A tensor to be activated.  
   - **Returns**:  
     - A new tensor with the activation applied.  
   - **Errors**:  
     - Memory allocation issues.

4. **`pub fn backward(ctx: *anyopaque, dValues: *Tensor(T)) !Tensor(T)`**  
   Computes the gradient of the activation function with respect to the input.  
   - **Parameters**:  
     - `dValues`: Gradients passed from the next layer.  
   - **Returns**:  
     - A tensor representing the gradients with respect to the layer input.  
   - **Errors**:  
     - Memory allocation failures.

5. **`pub fn printLayer(ctx: *anyopaque, choice: u8) void`**  
   Prints the activation type and any activation-specific parameters for debugging purposes.

6. **Other getters** (`get_n_inputs`, `get_n_neurons`, `get_input`, `get_input`) 


---

## poolingLayer.zig
This module implements a **Pooling layer**, which reduces spatial dimensions by applying operations such as Max, Min, or Average pooling.

### What is the point of pooling?
A pooling layer reduces the spatial dimensions of an input (such as height and width), helping to decrease the number of parameters and computations required. By taking operations like maximum or average values over small regions, pooling also provides a form of translation invariance and helps prevent overfitting.

### Mathematical explanation
The detailed mathematical explanation of the pooling operation is avaiable [here](#pooling-link)

### In Depth Function Documentation

#### `pub fn PoolingLayer(comptime T: type, poolType: PoolingType) type`
Initializes and returns a struct for pooling operations (Max, Min, or Average), implementing the `Layer` interface.

##### Methods

1. **`pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void`**  
   Prepares the layer with given pooling parameters (window size, stride).  

2. **`pub fn deinit(ctx: *anyopaque) void`**  
   Frees any internally allocated resources (if any).

3. **`pub fn forward(ctx: *anyopaque, input: *Tensor(T)) !Tensor(T)`**  
   Applies the selected pooling operation over the spatial dimensions of the input.  
   - **Parameters**:  
     - `input`: a tensor shaped `[batch_size, channels, height, width]`.  
   - **Returns**:  
     - A new tensor with reduced spatial dimensions depending on the pool size and stride.  
   - **Errors**:  
     - Allocation failures or shape mismatch.

4. **`pub fn backward(ctx: *anyopaque, dValues: *Tensor(T)) !Tensor(T)`**  
   Propagates gradients back.
   - **Parameters**:  
     - `dValues`: Gradient from the subsequent layer.  
   - **Returns**:  
     - A tensor of gradients with the shape of the original input.  
   - **Errors**:  
     - Memory allocation, shape mismatch.

5. **`pub fn printLayer(ctx: *anyopaque, choice: u8) void`**  
   Displays layer and pooling type  for debugging purposes.

6. **Other getters** (`get_n_inputs`, `get_n_neurons`, `get_input`, `get_input`) 


---

## batchNormLayer.zig
This module implements **Batch Normalization**, a technique to normalize activations to improve training speed and stability.

### What is the point of batch normalization?
A batch normalization layer normalizes the inputs within a mini-batch so that they have a consistent distribution. By doing so, it stabilizes and speeds up training, allowing for higher learning rates and reducing sensitivity to initialization. The layer also includes learnable parameters (gamma and beta) that preserve the representational power of the network.

### In Depth Function Documentation
#### `pub fn BatchNormLayer(comptime T: type) type`
Initializes and returns a struct for batch normalization.

##### Methods

1. **`pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void`**  
   Allocates parameters for gamma (scale), beta (shift), and possibly running mean/variance tracking.  
   - **Parameters**:  
     - `args`: include momentum, epsilon, and shape info.  
   - **Errors**:  
     - Memory allocation failures.

2. **`pub fn deinit(ctx: *anyopaque) void`**  
   Frees memory used by the parameters.

3. **`pub fn forward(ctx: *anyopaque, input: *Tensor(T)) !Tensor(T)`**  
   Normalizes the input over its batch dimension (and possibly other dims), then applies an operation og normalization.  
   - **Parameters**:  
     - `input`: Input tensor to be normalized.  
   - **Returns**:  
     - The normalized output tensor.  
   - **Errors**:  
     - Memory issues or shape mismatches.

4. **`pub fn backward(ctx: *anyopaque, dValues: *Tensor(T)) !Tensor(T)`**  
   Computes gradients for gamma/beta and adjusts them, while also calculating gradients passed to previous layers.  
   - **Parameters**:  
     - `dValues`: Incoming gradients from the subsequent layer.  
   - **Returns**:  
     - A gradient tensor shaped like the input.  
   - **Errors**:  
     - Memory allocation or shape mismatch errors.

5. **`pub fn printLayer(ctx: *anyopaque, choice: u8) void`**  
   Prints all possible parammeters for debugging purposes.

6. **Other getters** (`get_n_inputs`, `get_n_neurons`, `get_input`, `get_input`) 


---

## flattenLayer.zig
This module implements a **Flatten layer**, which reshapes multi-dimensional data (e.g., from convolutional layers) into a 2D shape suitable for Dense layers.

### In Depth Function Documentation

#### `pub fn FlattenLayer(comptime T: type) type`
Initializes and returns a struct that flattens input tensors.

##### Methods

1. **`pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void`**  
   Performs any necessary setup for flattening operations.  

2. **`pub fn deinit(ctx: *anyopaque) void`**  
   Deinitializes and frees any allocated resources.

3. **`pub fn forward(ctx: *anyopaque, input: *Tensor(T)) !Tensor(T)`**  
   Reshapes (flattens) the input from `[batch_size, dim1, dim2, ...]` to `[batch_size, dim1*dim2*...]`.  
   - **Parameters**:  
     - `input`: The multi-dimensional input tensor.  
   - **Returns**:  
     - A new tensor that is a flattened 2D representation of the input.  
   - **Errors**:  
     - Memory allocation or shape issues.

4. **`pub fn backward(ctx: *anyopaque, dValues: *Tensor(T)) !Tensor(T)`**  
   Restores the gradient into the original input shape before flattening.  
   - **Parameters**:  
     - `dValues`: The incoming gradient in flattened form.  
   - **Returns**:  
     - A gradient tensor reshaped to the original input dimensions.  
   - **Errors**:  
     - Shape mismatch or memory allocation failures.

5. **`pub fn printLayer(ctx: *anyopaque, choice: u8) void`**  
   Prints layer details relevant to flattening for debugging puposes.

6. **Other getters** (`get_n_inputs`, `get_n_neurons`, `get_input`, `get_input`) 

---

## denseLayer.zig
This module implements a fully connected (**Dense**) layer. It manages the layer-specific parameters (weights and biases) and provides the methods to perform forward and backward passes.

### What is the point of a dense layer?
A dense (fully connected) layer learns global patterns by considering all input features when producing each output neuron’s value. This layer performs an affine transformation, and is typically used near the end of a neural network to combine extracted features and make predictions.

### In Depth Function Documentation

#### `pub fn DenseLayer(comptime T: type) type`
Initializes and returns a struct that implements a fully connected layer.

##### Methods

1. **`pub fn init(ctx: *anyopaque, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void`**  
   Allocates and initializes the weight and bias tensors for the layer with random values.  
   - **Parameters**:  
     - `alloc`: A constant pointer to the memory allocator.  
     - `args`: Layer-specific initialization arguments (e.g., number of inputs, number of neurons).  
   - **Errors**:  
     - Memory allocation failures if the creation of weights/biases tensors fails.

2. **`pub fn deinit(ctx: *anyopaque) void`**  
   Deinitializes the layer, releasing any allocated memory.  

3. **`pub fn forward(ctx: *anyopaque, input: *Tensor(T)) !Tensor(T)`**  
   Computes the forward pass for the Dense layer.  
   - **Parameters**:  
     - `input`: A pointer to the input tensor.
   - **Returns**:  
     - A tensor that is the result of the matrix multiplication `input * weights` plus biases.  
   - **Errors**:  
     - Propagates tensor or memory allocation errors (e.g., shape mismatches, failed allocations).

4. **`pub fn backward(ctx: *anyopaque, dValues: *Tensor(T)) !Tensor(T)`**  
   Computes the gradients for weights and biases and produces the gradient to be passed to the previous layer.  
   - **Parameters**:  
     - `dValues`: The gradient flowing in from the subsequent layer.  
   - **Returns**:  
     - A tensor representing the gradient w.r.t. the layer’s inputs.  
   - **Errors**:  
     - Shape mismatch or memory errors (e.g., dimension mismatches, allocation failures).  

5. **`pub fn printLayer(ctx: *anyopaque, choice: u8) void`**  
   Prints layer-specific parameters, such as weights and biases for debugging purposes.

6. **Other getters** (`get_n_inputs`, `get_n_neurons`, `get_weights`, `get_bias`, `get_input`, `get_input`, `get_weightGradients`, `get_biasGradients`) 
