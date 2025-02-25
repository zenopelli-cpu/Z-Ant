# Tensor

## Overview

Tensors are multi-dimensional arrays that can represent scalars, vectors, matrices, and higher-dimensional data. A scalar is a 0-dimensional tensor, a vector is a 1-dimensional tensor, and a matrix is a 2-dimensional tensor. This can extend to even higher dimensions.

## Usage

Tensors are essential for representing and processing data in neural networks, enabling complex inputs like images, text sequences, or tabular data. Each neuron processes tensors through mathematical operations like matrix multiplication and weighted sums to compute outputs.

Common tensor applications in NN:

* Data Input: Images are represented as 3D tensors (height, width, color channels).
* Weights and Biases: Internal parameters (weights and biases) are tensors updated during training.
* Forward Propagation: Tensors are used to compute model outputs during inference by performing matrix operations.
* Optimization: Gradients are also tensors used to update model parameters during training.

## Tensor Class

The attributes of the class are as follows:

- **data**: Contains all the data of the tensor in a one-dimensional array.
- **size**: The dimension of the tensor, equal to `data.len`.
- **shape**: Defines the multidimensional structure of the tensor.
- **allocator**: Allocator used in the memory initialization of the tensor.

## Implemented Functions List

### Initializing/Deleting Methods:

- **init()**:
  - **Description**: Method used to initialize an undefined Tensor.
  - **Input**: `allocator: *const std.mem.Allocator → Constant pointer to a memory allocator`
  - **Output**: Instance of the current structure.

- **fromArray()**:
  - **Description**: Given a multidimensional array with its shape, returns the equivalent Tensor.
  - **Input**:
    - `allocator: *const std.mem.Allocator → Constant pointer to a memory allocator`
    - `inputArray: anytype → Generic array input of any type`
    - `shape: []usize → Slice defining the dimensions of the array.`
  - **Output**: New tensor.

- **copy()**:
  - **Description**: Returns a Tensor which is a copy of this Tensor (self).
  - **Input**: `self: *@This() → Mutable reference to the current structure instance`
  - **Output**: New tensor.

- **fromShape()**:
  - **Description**: Returns an all-zero tensor starting from the given shape.
  - **Input**:
    - `allocator: *const std.mem.Allocator → Constant pointer to a memory allocator`
    - `shape: []usize → Slice defining the dimensions (shape) of the tensor`
  - **Output**: New tensor.

- **toArray()**:
  - **Description**: Given the Tensor (self), returns the equivalent multidimensional array.
  - **Input**:
    - `self: @This() → Immutable reference to the current structure instance`
    - `comptime dimension: usize → Compile-time constant indicating the desired dimension of the output array`
  - **Output**: `!MagicalReturnType(T, dimension)` → A multidimensional array of type `T` with the specified dimension.

- **deinit()**:
  - **Description**: Frees all possible allocations. Use it every time you create a new Tensor.
  - **Input**: `self: *@This() → Mutable pointer to the current structure instance`
  - **Output**: `void`.

- **fill()**:
  - **Description**: Given any array and its shape, reshapes the tensor and updates `.data`.
  - **Input**:
    - `self: *@This() → Mutable reference to the current tensor instance`
    - `inputArray: anytype → The new array of data to fill the tensor`
    - `shape: []usize → Slice defining the new shape of the tensor`
  - **Output**: `void`.

### Getters and Setters:

- **getSize()**:
  - **Description**: Returns the size of a Tensor.
  - **Input**: `self: *@This() → Immutable reference to the current tensor instance`
  - **Output**: `usize → The size of the tensor`.

- **get()**:
  - **Description**: Given an index, returns `self.data[index]`.
  - **Errors**: `error.IndexOutOfBounds`
  - **Input**:
    - `self: *const @This() → Immutable reference to the current tensor instance`
    - `idx: usize → Index of the element to retrieve from the tensor data`
  - **Output**: Element at the specified index.

- **set()**:
  - **Description**: Sets `self.data[idx]` to a given value.
  - **Errors**: `error.IndexOutOfBounds`
  - **Input**:
    - `self: *@This() → Mutable reference to the current tensor instance`
    - `idx: usize → Index where the element should be set in the tensor data`
    - `value: T → The value of type `T` to set at the specified index`
  - **Output**: `void`.

- **get_at()**:
  - **Description**: Given the coordinates (indices), returns the corresponding value in the multidimensional array.
  - **Input**:
    - `self: *const @This() → Immutable reference to the current tensor instance`
    - `indices: []const usize → A slice of indices representing the position in the tensor`
  - **Output**: Element at the specified indices.

- **set_at()**:
  - **Description**: Given the value and the coordinates (indices), sets the value in the multidimensional array at the specified coordinates.
  - **Input**:
    - `self: *@This() → Mutable reference to the current tensor instance`
    - `indices: []const usize → A slice of indices representing the position in the tensor`
  - **Output**: `void`.

### Utils:

- **constructMultidimensionalArray()**:
  - **Description**: Converts `self.data` (monodimensional) and `self.shape` into an equivalent multidimensional array (recursive).
  - **Input**:
    - `allocator: *const std.mem.Allocator → Constant pointer to a memory allocator`
    - `ElementType: type → The type of the elements in the multidimensional array`
    - `data: []ElementType → Flat slice of the array's data`
    - `shape: []usize → Slice representing the shape of the array`
    - `depth: usize → Current depth level in the recursive construction`
    - `dimension: usize → Total number of dimensions of the multidimensional array`
  - **Output**: `!MagicalReturnType(ElementType, dimension - depth)` → The constructed multidimensional array with the specified dimensions.

- **MagicalReturnType()**:
  - **Description**: Generates a recursive multidimensional array type based on the specified element type and dimension count.
  - **Input**:
    - `DataType: type → The type of the elements in the multidimensional array`
    - `dim_count: usize → The number of dimensions for the array`
  - **Output**: A type representing a multidimensional array of the specified `DataType` and `dim_count` dimensions.

- **calculateProduct()**:
  - **Description**: Calculates the product of a slice.
  - **Input**: `slices: []usize → A slice of `usize` values representing dimensions or sizes`
  - **Output**: `usize → The product of all the values in the slice`.

- **flatten_index()**:
  - **Description**: Given the coordinates (indices) of a multidimensional Tensor, returns the corresponding position in the monodimensional space of `self.data`.
  - **Input**:
    - `self: *const @This() → Immutable reference to the current tensor instance`
    - `indices: []const usize → A slice representing the multidimensional indices to be converted`
  - **Output**: `!usize → The flattened index corresponding to the multidimensional indices`.

- **slice()**:
  - **Description**: Slices the current tensor based on the given `start_indices` and `slice_shape`, validates the slice, and returns a new tensor with the sliced data. If any errors occur, an error is returned.
  - **Input**:
    - `self: *Tensor(T) → Mutable reference to the current tensor instance`
    - `start_indices: []usize → A slice of starting indices for the slice operation`
    - `slice_shape: []usize → A slice defining the shape of the sub-tensor to be extracted`
  - **Output**: New tensor of type `T`.

- **copy_data_recursive()**:
  - **Description**: Recursive function to copy data.
  - **Input**:
    - `self: *Tensor(T) → Mutable reference to the current tensor instance`
    - `new_data: []T → The array where the sliced data will be stored`
    - `new_data_index: *usize → Mutable reference to the current index in `new_data`
    - `start_indices: []usize → Starting indices for the slice`
    - `slice_shape: []usize → Shape of the sliced sub-tensor`
    - `indices: []usize → Temporary array to track the current indices for recursion`
    - `dim: usize → Current depth level of recursion`
  - **Output**: `void`.

- **get_flat_index()**:
  - **Description**: Helper function to calculate the flat index from multi-dimensional indices.
  - **Input**:
    - `self: *Tensor(T) → Mutable reference to the current tensor instance`
    - `indices: []usize → A slice representing the multidimensional indices`
  - **Output**: `!usize → The flattened index corresponding to the multidimensional indices`.

- **getStrides()**:
  - **Description**: Function to calculate strides for the tensor.
  - **Input**: `self: *Tensor(T) → Mutable reference to the current tensor instance`
  - **Output**: `![]usize → A slice representing the strides for each dimension`.

- **setToZero()**:
  - **Description**: Sets all tensor values to zero.
  - **Input**: `self: *@This() → Immutable reference to the current tensor instance`
  - **Output**: `void`.

- **gather()**:
  - **Description**: Gather elements from the tensor along an axis using the provided indices. The `axis` parameter specifies the axis along which the elements will be gathered. The `indices` tensor must have the same number of dimensions as the input tensor, except for the axis dimension. The shape of the output tensor is the same as the shape of the indices tensor, with the axis dimension removed. The output tensor is created by copying elements from the input tensor using the indices tensor.
  - **Input**:
    - `self: *@This() → Immutable reference to the current tensor instance`
    - `indices: Tensor(usize) → Tensor of indices that specifies the elements to gather`
    - `selected_axis: isize → The axis along which to gather data`
  - **Output**: `!@This() → A new tensor with the gathered data`.

- **slice_onnx()**:
  - **Description**: Implements the ONNX slice operator (https://onnx.ai/onnx/operators/onnx__Slice.html). Takes a tensor and extracts a slice along multiple axes. `starts`: Starting indices for each axis. `ends`: Ending indices for each axis (exclusive). `axes`: Which axes to slice (if null, assumes [0, 1, 2,...]). `steps`: Step sizes for each axis (if null, assumes all 1s).
  - **Input**:
    - `self: *Tensor(T) → Mutable reference to the tensor instance`
    - `starts: []const i64 → Starting indices for each axis`
    - `ends: []const i64 → Ending indices for each axis`
    - `axes: ?[]const i64 → Optional axis selection for slicing`
    - `steps: ?[]const i64 → Optional step sizes for each axis`
  - **Output**: `!Tensor(T) → A new tensor containing the sliced data`.

- **ensure_4D_shape()**:
  - **Description**: Ensures the input shape is 4D by padding with 1s if necessary. Returns an error if the shape has more than 4 dimensions.
  - **Input**: `shape: []const usize → The input shape of the tensor as a list of dimension sizes`
  - **Output**: `![]usize → A padded 4D shape array`.

- **info_metal()**:
  - **Description**: Bare metal version of tensor info that uses a logging function instead of `std.debug.print`.
  - **Input**: -
  - **Output**: Logs the tensor's size, shape, and data.

- **intToString()**:
  - **Description**: Helper function for integer-to-string conversion.
  - **Input**:
    - `value: usize → The integer value to convert to a string`
    - `buffer: []u8 → A mutable byte array where the resulting string will be stored`
  - **Output**: `usize → The number of characters written to the buffer`.

- **floatToString()**:
  - **Description**: Helper function for float-to-string conversion.
  - **Input**:
    - `value: f32 → The floating-point value to convert to a string`
    - `buffer: []u8 → A mutable byte array where the resulting string will be stored`
  - **Output**: `usize → The number of characters written to the buffer`.

- **flattenArray()**:
  - **Description**: Recursive function to flatten a multidimensional array.
  - **Input**:
    - `T: type → The type of the elements in the array`
    - `arr: anytype → The multidimensional array that will be flattened`
    - `flatArr: []T → A mutable array where the flattened array will be stored`
    - `startIndex: usize → The index in flatArr where the flattening will start`
  - **Output**: `usize → The updated index in the flattened array after flattening`.

### Prints:

- **info()**:
  - **Description**: Prints all the possible details of a tensor.
  - **Input**: `self: *@This() → Immutable reference to the current tensor instance`
  - **Output**: `void`.

- **print()**:
  - **Description**: Prints all the array `self.data` in an array.
  - **Input**: `self: *@This() → Immutable reference to the current tensor instance`
  - **Output**: `void`.

- **printMultidim()**:
  - **Description**: Prints the `Tensor()` in a more readable way (uses `_printMultidimHelper()`).
  - **Input**: `self: *@This() → Immutable reference to the current tensor instance`
  - **Output**: `void`.

- **_printMultidimHelper()**:
  - **Description**: Prints the opening bracket with a number of tabs equal to `idx`.
  - **Input**:
    - `self: *@This() → Immutable reference to the current tensor instance`
    - `offset: usize → The current offset for accessing the tensor data`
    - `idx: usize → The current depth level for recursive printing`
  - **Output**: `void`.
