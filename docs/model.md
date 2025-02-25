# Model
This module provides the structure for the creation and management of multiple layers neural networks models.  
It supports adding layers, performing forward and backward pass and memory management.
## In depth Function documentation :
## model.zig
The main component of this module is the function :  
`pub fn Model(comptime T: type) type`  
The only input parameter of this function is the type that will be used in the definition of the struct and its functions.  
This function returns the struct type that represents the neural network model, which contains the methods for the model management.

The methods are:

1. `pub fn init(self: *@This()) !void`  
   Initializes the model, setting up an empty list of layers and initializing the input tensor  
   Errors:
   - Returns an error if memory allocation for the `layers` array or `input_tensor` fails.

2. `pub fn deinit(self: *@This()) void`  
   Deinitializes the model, releasing memory for each layer and the input tensor.  
   This method iterates through each layer, deinitializes it, and then frees the layer array and input tensor memory.

3. `pub fn addLayer(self: *@This(), new_layer: layer.Layer(T)) !void`  
   Adds a new layer to the model.  
   Parameters:
   - `new_layer`: A pointer to the new layer to add to the model  
   
   Errors:  
   - Returns an error if reallocating the `layers` array fails.

4. `pub fn forward(self: *@This(), input_tensor: *tensor.Tensor(T)) !*tensor.Tensor(T)`  
   Executes the forward pass through the model with the specified input tensor.  
   Parameters:
   - `input_tensor`: A pointer to the input tensor.
   
   Returns:
   - a pointer to the output tensor of the model after the forward pass.

   Errors:
   - Returns an error if any layer's forward pass or tensor copying fails.

5. `pub fn backward(self: *@This(), gradient: *tensor.Tensor(T)) !void`  
   Executes the backward pass through the model with the specified gradient tensor.  
   Parameters:
   - `gradient`: A pointer to the gradient tensor to backpropagate.

   Errors:
   - Returns an error if any layer's backward pass or tensor copying fails.

6. `fn getPrevOut(self: *@This(), layer_numb: usize) *tensor.Tensor(T)`  
   Retrieves the output of the specified layer or the input tensor for the first layer.  
   Parameters:
   - `layer_numb`: The index of the layer whose output tensor is to be retrieved.

   Returns:
   - A pointer to the output tensor of the specified layer, or the input tensor if `layer_numb` is zero.

   Errors:
   - Returns an error if the index is out of bounds or other tensor-related errors occur.


---


# lossFunction.zig
This module defines the loss functions implemented so far, which are used to compare the model's predictions with the target, i.e. the actual expected values, and calculate the error.

As of now, the implemented Loss Functions are:
- Mean-Square-Error (MSE) : it measures the average squared difference between the predicted values and the true values (targets). MSE penalizes larger errors more than smaller ones due to the squaring of differences, making it sensitive to outliers.
- Categorical Cross-Entropy (CCE), also known as Softmax Loss : it measures the difference between the true label distribution and the predicted probability distribution. Often used for multi-class classification problems.

The possible Types of Loss Function are defined in the **LossType** enum. Again, as of now, the Types are MSE and CCE.
### LossFunction interface
This interface is used to initialize a generic loss function:  
`pub fn LossFunction(lossType: LossType) type`  
Parameters:
- `lossType`: the needed LossType

Returns:
- A struct type containing the loss function having `lossType` Type

The included methods are:
1. `pub fn computeLoss(self: *const @This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T)`  
   Abstract Method called to compute the loss of a prediction.  
   Parameters:
   - `T`: is a comptime type parameter that represents the data type of the elements in the tensors.
   - `predictions`: is the pointer to the predictions tensor produced by the model
   - `targets`: is the target tensor, or actual values.

   Returns:
   - a tensor where each element represents the computed loss (e.g. MSE or CCE) for a corresponding row in the predictions tensor when compared to the targets.  

   Errors:
   - can propagate errors generated during the execution of the specific computeLoss function.

2. `pub fn computeGradient(self: *const @This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T)`  
   Abstract Method called to compute the gradient of the loss of a prediction.  
   Parameters:
   - `T`: is a comptime type parameter that represents the data type of the elements in the tensors.
   - `predictions`: is the pointer to the predictions tensor produced by the model
   - `targets`: is the target tensor, or actual values.

   Returns:
   - a tensor where each element represents the computed gradient of the loss. Its elements represent the rate of change of the loss with respect to each corresponding prediction.

   Errors:
   - can propagate errors generated during the execution of the specific computeLoss function.

### MSELoss
`pub fn MSELoss() type` implements the LossFunction interface for the MSE, of course.

Methods:

1. `computeLoss`, parameters and result as described in the interface  
   Errors:
   - `TensorMathError.InputTensorDifferentSize` : returned if predictions and targets have different sizes
   - `TensorMathError.MemError` : returned on tensors' allocation failure.
   - can propagate the errors generated by `basicChecks` function (described below)

2. `computeGradient`, parameters and result as described in the interface  
   Errors:
   - `LossError.SizeMismatch` : returned if predictions and targets have different sizes or shapes
   - can propagate the errors generated by `basicChecks` function (described below)

3. `fn multidim_MSE(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T), out_tensor: *Tensor(T), current_depth: usize, location: []usize) !void`  
   Accessory function used in computeLoss.  
   Method used to handle multidimensionality in tensors.
   This function recursively computes the Mean Squared Error across the multiple dimensions of a tensor. It traverses the dimensions of predictions and targets, summing up squared differences along the last dimension, and stores the result in out_tensor.  

   Parameters:
   - `T` : The data type for tensor elements (e.g., f32 or f64).
   - `predictions` : as always, it is the pointer to the tensor containing model predictions.
   - `targets` : the pointer to the tensor containing target values.
   - `out_tensor` : pointer to the tensor where the computed MSE values will be stored.
   - `current_depth` : The current depth in the recursion, tracking which dimension is being processed.
   - `location` : An array storing the coordinates of the current tensor position being processed.

   Errors:
   - an error is returned if memory allocation or tensor access fails

   Some key Points:
   - Uses recursion to navigate through multiple dimensions
   - Memory allocation (alloc and free) is used to store intermediate locations.
   - Handles integer and floating-point division correctly based on T's type.

### CCELoss
`pub fn CCELoss() type` implements the LossFunction for CCE.

Methods:
1. `computeLoss`, parameters and result as described in the interface, and errors as described in MSELoss.

2. `computeGradient`, parameters and result as described in the interface, and errors as described in MSELoss.

3. `fn multidim_CCE(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T), out_tensor: *Tensor(T), current_depth: usize, location: []usize) !void`  
   Accessory function called by the computeLoss function defined in the CCELoss struct.  
   This function computes the Categorical Cross-Entropy (CCE) loss for multi-dimensional tensors. It recursively processes the tensor along each dimension until it reaches the final dimension, where the loss calculation occurs.

   Parameters:
   - `T` : The data type for tensor elements (e.g., f32 or f64).
   - `predictions` : as always, it is the pointer to the tensor containing model predictions.
   - `targets` : the pointer to the tensor containing target values.
   - `out_tensor` : pointer to the tensor where the computed CCE values will be stored.
   - `current_depth` : The current depth in the recursion, tracking which dimension is being processed.
   - `location` : An array storing the coordinates of the current tensor position being processed.

`fn basicChecks(comptime T: anytype, tensor: *Tensor(T)) !void`  
This function performs fundamental validity checks on a tensor to ensure it is well-formed before further processing in `computeLoss`.  
Checks:
- Not empty data: verifies tensor.data.len is greater than zero and not NaN.
- Not zero shape: verifies tensor.shape.len is greater than zero and not NaN.
- Valid size: verifies tensor.size is greater than zero and not NaN.

Errors:
- TensorError.EmptyTensor: If data.len == 0 or shape.len == 0.
- TensorError.ZeroSizeTensor: If size == 0.


---


# optim.zig
This module provides **optimization algorithms** for training machine learning models and is responsible for updating model parameters during training by applying gradient-based optimization techniques.

As of now, the optimizers that have been defined are:
- **SGD** (Stochastic Gradient Descent), whose primary goal is to identify the model parameters that provide the maximum accuracy on both training and test datasets. The gradient is a vector pointing in the general direction of the function’s steepest rise at a particular point. The algorithm might gradually drop towards lower values of the function by moving in the opposite direction of the gradient, until reaching the minimum of the loss function.
- The **Adam** optimizer (not fully implemented), short for “Adaptive Moment Estimation,” an iterative optimization algorithm used to minimize the loss function.

Two enums define the pooling function types and the implemented optimizers:

`PoolingType` Enum  
Defines different pooling operations: `Max`, `Min`, and `Avg`.

`Optimizers` Enum  
Lists available optimizers: `SGD`, `Adam`, and `RMSprop` (this last one is not implemented actually).

### Optimizers documentation:  
`pub fn Optimizer(comptime T: type, comptime Xtype: type, comptime YType: type, func: fn (comptime type, comptime type, comptime type, f64) type, lr: f64) type`  
This is the generic function to create an optimizer.  
Uses a provided function func to initialize an optimizer struct type.  
Parameters:  
   - `T` : the data type used for tensor values in the model (e.g. weights, biases, gradients).
   - `XType` : the type of the input data fed into the model.
   - `YType` : the output type of the model.
   - `func` : the step optimizer's step function.
   - `lr` : is the learning rate.  
   note: as of now, XType and YType are not used in the implementations.

Returns:
   - a struct type containing a step function, which calls the optimizer's step function.  
      `step` function signature : `pub fn step(self: *@This(), model: *Model.Model(T)) !void`

### SGD Optimizer
`pub fn optimizer_SGD(T: type, XType: type, YType: type, lr: f64) type`  
Implements the SGD optimizer.  

Parameters: as described in the generic function `Optimizer`, nut without the parameter `func`.

Returns: sgd optimizer struct type.

Methods:
1. `pub fn step(self: *@This(), model: *Model.Model(T)) !void`  
   Loops through the layers of the model and updates their weights and biases using SGD via the helper function `update_tensor`.

   Parameters:
   - `model` : model is a pointer to an instance of the struct returned by Model(T), which represents the model.

   Errors:
   - can propagate errors generated by the called functions, such as `update_tensor`.

2. `fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void`  
   It's an helper function which performs the actual weight update for each parameter tensor.

   Parameters:
   - `t` : the tensor on which the step is applied.
   - `gradients` : the gradients' tensor.

   Errors:
   - `TensorMathError.InputTensorDifferentSize` : if `t` and `gradients` have different sizes.


### Adam Test Optimizer
`pub fn optimizer_ADAMTEST(T: type, lr: f64) type`  
Implements the Adam test optimizer.
NOTE: not fully implemented.  

Parameters:
- T : the data type used for tensor values in the model (e.g. weights, biases, gradients).
- lr : the learning rate.

Returns: adam test optimizer struct type.

Methods (and relative paramters, return value and errors): same as in `optimizer_SGD`.


---


# layer.zig
This module contains the definition of the layers that can be used in the neural network.  
There are function to initialize random weigths. Initialization right now is completely random but in the future it will possible to use proper initialization techniques.  
Layer can be stacked in a model and they implement proper forward and backward methods.

The Layer Types are enumerated in the `LayerType` enum:
- DenseLayer
- DefaultLayer
- ConvolutionalLayer
- PoolingLayer
- ActivationLayer
- FlattenLayer
- BatchNormLayer

Methods:
- `pub fn randn(comptime T: type, allocator: *const std.mem.Allocator, n_inputs: usize, n_neurons: usize) ![]T`  
  Initializes a vector of random values with a normal distribution.  
  Parameters:
  - `T`: data type.
  - `allocator`: the memory allocator.
  - `n_inputs`: number of inputs.
  - `n_neurons`: number of neurons.

  Returns: a dynamically allocated vector of randomly initialized weights.

  Errors:
  - could return errors if the vector allocation fails.

- `pub fn zeros(comptime T: type, allocator: *const std.mem.Allocator, n_inputs: usize, n_neurons: usize) ![]T`  
  Initializes a vector of zeros used for bias.  
  Pay attention, when using Tensor.fromShape() already initializes a tensor of zeros. Do not use zeros() and then Tensor.fromArray() because is redundant.  

  Parameters:
  - `T` : Data type.
  - `allocator` : Memory allocator.
  - `n_inputs` : Number of inputs.
  - `n_neurons` : Number of neurons.

  Returns: a dynamically allocated vector of zeros.

  Errors:
  - could return errors if the vector allocation fails.

### Interface Layer
Layer() is the superclass for all the possible implementation of a layer (Activation, Dense, Conv ... see /Layer folder).

`pub fn Layer(comptime T: type) type`  
Parameter:
- `T` : comptime type of the values in the layer

Returns:
- A struct representing a neural network layer

Methods:
- `pub fn init(self: Self, alloc: *const std.mem.Allocator, args: *anyopaque) anyerror!void`  
  Initializes the layer with given parameters.

  Parameters:
  - `alloc` : constant pointer to the memory allocator.
  - `args`

  Errors:
  - returns error if the initialization fails due to memory allocation errors.

- `pub fn deinit(self: Self) void`  
  Deallocates the layer, ensuring proper memory management.

- `pub fn forward(self: Self, input: *Tensor(T)) !Tensor(T)`  
  Computes forward propagation for the layer.

  Parameters:
  - `input` : the input tensor

  Returns:
  - the output tensor of the forward pass

- `pub fn backward(self: Self, dValues: *Tensor(T)) !Tensor(T)`  
  Computes backward propagation (gradient computation).

  Parameters:
  - `dValues` : the gradient tensor

- `pub fn printLayer(self: Self, choice: u8) void`  
  Prints details about the layer.
  
  Parameters:
  - `choice` : defines the printing format for the layer (see /Layer folder).

- `pub fn get_n_inputs(self: Self) usize`  
  Returns the number of input neurons.

- `pub fn get_n_neurons(self: Self) usize`  
  Returns the number of neurons in the layer.

- `pub fn get_input(self: Self) *const Tensor(T)`  
  Returns a constant pointer to the input tensor.

- `pub fn get_output(self: Self) *Tensor(T)`  
  Returns a constant pointer to the output tensor.
