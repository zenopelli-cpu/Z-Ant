# Data loader 

## Overview

This module provides a single struct that contains all the methods in the file:

The struct is constructed through a method, and is defined by three comptime parameters:
- __OutType__: `type` | type of the parameters to be loaded.
- __Ftype__: `type` | type of the features.
- __LabelType__: `type` | type of the labels.

And two other parameters:
- __batchSize__: `i16` | number of batches to load.
- __dimInput__: 'usize' | size in bytes of the input data.

```zig
fn DataLoader(comptime OutType: type, comptime Ftype: type, comptime LabelType: type, batchSize: i16, dimInput usize)
```
## __Internal struct fields__:

- __X__: `MagicalReturnType(OutType, dimInput)`
- __y__: `[]OutType`
- __x_index__: `usize`
- __y_index__: `usize`
- __xTensor__: `tensor.Tensor(OutType)`
- __yTensor__: `tensor.Tensor(OutType)`
- __batchSize__: `usize`
- __XBatch__: `MagicalReturnType(OutType, dimInput)`
- __yBatch__: `[]OutType`
- __XBuffer__: `?[]OutType`
- __X_train__: `?MagicalReturnType(OutType, dimInput)`
- __y_train__: `?[]OutType`
- __X_test__: `?MagicalReturnType(OutType, dimInput)`
- __y_test__: `?[]OutType`
- __x_train_index__: `usize`
- __y_train_index__: `usize`
- __x_test_index__: `usize`
- __y_test_index__: `usize`

## __Struct functions__:

- __xNext()__:
    - **Description**: Moves ahead an iterator on X, returns an optional array of the features corrisponding to it. `null` if the iterator is already at the end of X before invoking the method.
    - **Input**: `none`
    - **Output**: `?[]Ftype`.

- __yNext()__:
    - **Description**: Moves ahead an iterator on Y, returns an optional lavel corrisponding to its position. `null` if the iterator is already at the end of Y before invoking the method.
    - **Input**: `none`
    - **Output**: `?LabelType`.

- __toTensor()__:
    - **Description**: Converts current state of the struct to Tensors of the shapes currently passed as an argument.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Allocator used for initializing the Tensors`
        - `shapeX: *[]usize -> Desired shape of the tensor created from X`
        - `shapeY: *[]usize -> Desired shape of the tensor created from Y`
    - **Output**: `!void`.

- __reset()__:
    - **Description** : Resets the index of all iterators.
    - **Input**: `none`
    - **Output**: `void`.

- __xNextBatch()__:
    - **Description**: Attempts to retrieve the next batch of data from the source and returns it, returns null if its run out of data to read.
    - **Input**: `none`
    - **Output**: `?[][]OutType`.

- __yNextBatch()__:
    - **Description**: Attempts to retrieve the next batch of labels from Y, can return null if theres no more to read.
    - **Input**: `none`
    - **Output**: `?[]OutType`.

- __trainTestSplit()__:
    - **Description**: Splits a percentage of the data in x and Y and copies the features and labels into respective arrays `X_train`, `y_train`, `X_test`, `y_test` allocated automatically by the function.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Constant pointer to a memory allocator`
        - `perc: f32 -> Percentage of data to copy`
    - **Output**: `!void`.

- __shuffle()__:
    - **Description**: Shuffle the data using the Knuth shuffle algorithm.
    - **Input**:
        - `rng: std.Random.DefaultPrng -> Random number generator`
    - **Output**: `void`.

- __xTrainNextBatch()__:
    - **Description**: Loads a batch of size `batchSize` in the respective array for the Trainer methods to work on.
    - **Input**: `none`
    - **Output**: `?MagicalReturnType(OutType, dimInput) -> A copy of the batch that was just loaded`.

- __yTrainNextBatch()__:
    - **Description**: Loads a batch of size `batchSize` in the respective array for the Trainer methods to work on.
    - **Input**: `none`
    - **Output**: `?[]OutType -> A copy of the labels that were just loaded`.

- __xTestNextBatch()__:
    - **Description**: Loads a batch of size `batchSize` in the respective array for it to be tested.
    - **Input**: `none`
    - **Output**: `?MagicalReturnType(OutType, dimInput) -> A copy of the batch that was just loaded`.

- __yTestNextBatch()__:
    - **Description**: Loads a batch of size `batchSize` in the respective array for it to be tested.
    - **Input**: `none`
    - **Output**: `?[]OutType -> A copy of the labels that were just loaded`.

- __deinit()__:
    - **Description**: Deinits and deallocates the entire struct.
    - **Input**: `allocator: * const std.mem.Allocator -> Pointer to allocator`
    - **Output**: `void`.

- __fromCSV()__:
    - **Description**: Loads data from a CSV-like file.
    - **Input**: 
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `filePath: []const u8 -> System path of the csv file`
        - `featureCols: []const usize -> Array of indexes of features to load`
        - `labelCol: usize -> Index of the labels column`
    - **Output**: `!void`.

- __loadMNISTImages()__:
    - **Description**: Load data from a MNIST dataset.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `filePath: []const u8 -> System path to the MNIST dataset`
    - **Output**: `!void`.

- __loadMNISTImages2D()__:
    - **Description**: Load data from a 2d MNIST dataset.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `filePath: []const u8 -> System path to the MNIST dataset`
    - **Output**: `!void`.

- __loadMNISTImages2DStatic()__:
    - **Description**: Static version of previous function.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `filePath: []const u8 -> System path to the MNIST dataset`
        - `numImages: usize -> Number of images to load`
        - `numRows: usize -> Number of rows`
        - `numCols: usize -> Number of columns`
    - **Output**: `!void`.

- __loadMNISTLabels()__:
    - **Description**: Load labels from MNIST dataset
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `filePath: []const u8 -> System path to the MNIST dataset`
    - **Output**: `!void`.

- __loadMNISTDataParallel()__:
    - **Description**: Load data in parallel from a MNIST dataset, allows for labels and data to be in separate paths.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `imageFilePath: []const u8 -> System path to the MNIST image dataset`
        - `labelFilePath: []const u8 -> System path to the dataset Labels`
    - **Output**: `!void`.

- __loadMNIST2DDataParallel()__:
    - **Description**: Load data in parallel from a 2D MNIST dataset, allows for labels and data to be in separate paths.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `imageFilePath: []const u8 -> System path to the MNIST image dataset`
        - `labelFilePath: []const u8 -> System path to the dataset Labels`
    - **Output**: `!void`.

- __loadImages()__:
    - **Description**: Load image from MNIST dataset.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `imageFilePath: []const u8 -> System path to the MNIST dataset`
    - **Output**: `!void`.

- __loadImages2D()__:
    - **Description**: Load images from 2D MNIST dataset.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `imageFilePath: []const u8 -> System path to the MNIST dataset`
    - **Output**: `!void`.

- __loadLabels()__:
    - **Description**: Load labels from MNIST dataset.
    - **Input**:
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `labelFilePath: []const u8 -> System path to the MNIST dataset`
    - **Output**: `!void`.

- __readCSVLine()__:
    - **Description**: Read a line from a CSV file.
    - **Input**:
        - `reader: *std.fs.File.Reader -> Pointer to reader`
        - `lineBuf: []u8 -> Buffer to be allocated by the user`
    - **Output**: `!?[]u8`.

- __splitCSVLine()__:
    - **Description**: Splits a CSV line.
    - **Input**:
        - `line: []u8 -> Pointer to char array`
        - `allocator: *const std.mem.Allocator -> Pointer to allocator`
    - **Output**: `![]const []u8`.

- __freeCSVColums()__:
    - **Description**: Frees the columns of a CSV file.
    - **Input**:
    - **Output**:

- __parseXType()__:
    - **Description**: Parses a comptime parameter for X and returns its value if possible, an error if not.
    - **Input**:
        - `comptime XType: type -> Type to try and infer`
        - `self: []const u8 -> Bytes to try and read the value from`
    - **Output**: `!XType`

- __parseYType()__:
    - **Description**: Parses a comptime parameter for y and returns its value if possible, an error if not.
    - **Input**:
        - `comptime YType: type -> Type to try and infer`
        - `self: []const u8 -> Bytes to try and read the value from`
    - **Output**: `!Ytype`

# Data processor

## Overview

Data processor functions are used in combination with Data Loader in the Trainer methods, they encompass Normalization operations in different formats 

### __Data processor functions__

- __normalize()__:
    - **Description**: Normalizes a Tensor of T types according to the chosen type.
    - **Input**:
        - `comptime T: anytype -> Type of data in the tensor`
        - `tensor: *Tensor(T) -> Tensor to normalize`
        - `normalizationType: NormalizationType -> Type of normalization requested`
    - **Output**: `!void`.

- __multidimNormalizeUnityBased()__:
    - **Description**: Normalizes each row in a multidim. Tensor.
    - **Input**:
        - `comptime T: anytype -> Type composing the Tensor`
        - `tensor: *Tensor(T) -> Tensor to normalize`
    - **Output**: `!void`.

- __multidimNormalizeStandard()__:
    - **Description**: Normalizes a given Tensor.
    - **Input**:
        - `comptime T: anytype -> Type composing the Tensor`
        - `tensor: *Tensor(T) -> Tensor to normalize`
    - **Output**: `!void`.

# Trainer

## Overview

Trainer contains the higher level functions related to training and properly configuring a Data Loader or a Tensor, most users should and will interact with this top layer more often, and as such must be treated with care,

### __Trainer functions()__

- __TrainDataLoader()__:
    - **Description**: Trains a Dataloader using its data, goes through a number of set epochs and then runs the trained loader through a number of validation steps, to then reset the loader and print stats about the process that it just underwent.
    - **Input**:
        - `comptime T: type -> Type of data in the output tensors and loader`
        - `comptime XType: type -> Feature type for the loader, input`
        - `comptime YType: type -> Label type for the loader, output`
        - `comptime allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `comptime batchSize: i16 -> Size of batches to load`
        - `features: usize -> Number of features for the loader`
        - `model: *Model(T, allocator) -> Pointer to model`
        - `load: *DataLoader(T, Xtype, YType, BatchSize) -> Data loader to train`
        - `epochs : u32 -> Number of epochs to train for`
        - `comptime lossType: LossType -> Type of Loss function to use` 
        - `comptime lr: f64 -> Set learning rate`
        - `training_size: f32 -> Percentage of training set to split from loader`
    - **Output**: `!void`.

- __TrainDataLoader2D()__:
    - **Description**: Trains a 2D Dataloader using its data, goes through a number of set epochs and then runs the trained loader through a number of validation steps, to then reset the loader and print stats about the process that it just underwent.
    - **Input**:
        - `comptime T: type -> Type of data in the output tensors and loader`
        - `comptime XType: type -> Feature type for the loader, input`
        - `comptime YType: type -> Label type for the loader, output`
        - `comptime allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `comptime batchSize: i16 -> Size of batches to load`
        - `features: usize -> Number of features for the loader`
        - `model: *Model(T, allocator) -> Pointer to model`
        - `load: *DataLoader(T, Xtype, YType, BatchSize) -> Data loader to train`
        - `epochs : u32 -> Number of epochs to train for`
        - `comptime lossType: LossType -> Type of Loss function to use` 
        - `comptime lr: f64 -> Set learning rate`
        - `training_size: f32 -> Percentage of training set to split from loader`
    - **Output**: `!void`.

- __computeAccuracy()__:
    - **Description**: Compute accuracy of model by comparing to a supervised accurate prediction set.
    - **Input**:
        - `comptime T: type -> Type of data in the tensor`
        - `predictions: *Tensor.Tensor(T) -> Pointer to tensor containing the predictions`
        - `targets: *Tensor.Tensor(T) -> Pointer to tensor containing model predictions to evaluate`
    - **Output**: `!u16 -> Number of correct predictions`.

- __convertToOneHot()__:
    - **Description**: Converts a given Tensor into a One Hot.
    - **Input**: 
        - `comptime T: type -> Type of data in the tensor`
        - `batchSize: i16 -> Size of the batches`
        - `yBatch: *Tensor.Tensor(T) -> Pointer to tensor to convert`
    - **Output**: `!void`.

- __trainTensors()__:
    - **Description**: Trains given tensors using the given input tensors.
    - **Input**:
        - `comptime T: type -> Type of tensors and model`
        - `comptime allocator: *const std.mem.Allocator -> Pointer to allocator`
        - `model: *Model(T) -> Pointer of Model struct`
        - `input: *Tensor.Tensor(T) -> Pointer to input tensor`
        - `targets: *Tensor.Tensor(T) -> Pointer to target tensors`
        - `epochs: u32 -> Number of epochs to train through`
        - `comptime lr: f64 -> learning rate`
    - **Output**: `!void`.

- __print_start_training()__:
    - **Description**: Prints training start splash.
    - **Input**: `none`
    - **Output**: `void`.

- __print_end_training()__:
    - **Description**: Prints training end splash.
    - **Input**: `none`
    - **Output**: `void`.

- __clipGradients()__:
    - **Description**: Helper function for gradient clipping, used by other functions.
    - **Input**:
        - `comptime T: anytype -> Type composing the Tensor`
        - `tensor: *Tensor(T) -> Tensor to normalize`
        - `max_norm: T -> Max value to be left without being clipped`
    - **Output**: `!void`.

