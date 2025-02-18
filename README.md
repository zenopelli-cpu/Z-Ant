# Z-Ant
![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)
## Project Overview

**Zant** (Zig-Ant) is an open-source SDK designed to simplify deploying Neural Networks (NN) on microcontrollers. Written in Zig, Zant prioritizes cross-compatibility and efficiency, providing tools to import, optimize, and deploy NNs seamlessly, tailored to specific hardware.

### Why Zant?

1. Many microcontrollers (e.g., ATMEGA, TI Sitara) lack robust deep learning libraries.
2. No open-source solution exists for end-to-end NN optimization and deployment.
3. Inspired by cutting-edge research (e.g., MIT Han Lab), we leverage state-of-the-art optimization techniques.
4. Collaborating with institutions like Politecnico di Milano to advance NN deployment on constrained devices.
5. Built for flexibility to adapt to new hardware without codebase changes.

### Key Features

- **Optimized Performance:** Supports quantization, pruning, and hardware acceleration (SIMD, GPU offloading).
- **Efficient Memory Usage:** Incorporates memory pooling, static allocation, and buffer optimization.
- **Cross-Platform Support:** Works on ARM Cortex-M, RISC-V, and more.
- **Ease of Integration:** Modular design with clear APIs, examples, and documentation.

### Use Cases

- **Real-Time Applications:** Object detection, anomaly detection, and predictive maintenance on edge devices.
- **IoT and Autonomous Systems:** Enable AI in IoT, drones, robots, and vehicles with constrained resources.

---

## Roadmap & Timeline

We are actively working towards achieving key milestones for Zant's development and optimization:

### **Short-Term Goals (Q1 2025)**

- **March 5, 2025:** Run MNIST inference on a Raspberry Pi Pico 2, importing models from ONNX.
- **April 30, 2025:** Get YOLO running efficiently on Raspberry Pi Pico 2.

### **Mid-Term Goals (Q2-Q3 2025)**

- Transition to a **Shape Tracker**, optimizing tensor operations.
- Develop a **frontend interface** for easier interaction with the library.
- Implement **im2tensor**, converting JPEG/PNG images into our optimized tensor format.
- Optimize **code generation**, allowing users to choose between **flash storage** or **RAM** for execution.
- Expand **ONNX operation support**, ensuring compatibility with more layers and architectures.

### **Long-Term Goals (Q3 2025)**

- Begin work on **pruning and quantization** to improve network efficiency.
- Support additional **microcontrollers and architectures**.
- Develop **benchmarking tools** for profiling model execution on constrained devices.
- Improve the **compiler backend**, integrating more advanced optimization passes.
- Enhance support for **real-time inference**, focusing on low-latency applications.

---

## Getting Started

### Prerequisites

1. Install the [latest Zig compiler](https://ziglang.org/learn/getting-started/).
2. Brush up your Zig skills with [Ziglings exercises](https://codeberg.org/ziglings/exercises).

### Running the Project

Navigate to the project folder and execute:

```sh
zig build run
```

### Testing

1. Add new test files to `build.zig/test_list` if not already listed.
2. Run:
   ```sh
   zig build
   zig build test --summary all
   ```
   *(Ignore stderr warnings.)*

To run all tests, including computationally heavy ones, use:

```sh
zig build test -Dheavy --summary all
```

### Documentation

Generated using [Zig's standard documentation format](https://ziglang.org/documentation/master/#Doc-Comments).

---

## Using the Library

To use Zant effectively, you can create and train neural networks using a structured approach. Below is an example of how to define and configure a simple model.

### **1. Initialize the Model**

```zig
const allocator = @import("pkgAllocator").allocator;
var model = Model(f64){
    .layers = undefined,
    .allocator = &allocator,
    .input_tensor = undefined,
};
try model.init();
```

### **2. Add Layers to the Model**

#### **Convolutional Layer**

```zig
var conv1 = ConvolutionalLayer(f64){
    .input_channels = 1,
    .kernel_shape = .{ 32, 1, 3, 3 },
    .stride = .{ 1, 1 },
    .allocator = &allocator,
};
var conv1_layer = conv1.create();
try conv1_layer.init(&allocator, @constCast(&conv1));
try model.addLayer(conv1_layer);
```

#### **Activation Layer (ReLU)**

```zig
var conv1_activ = ActivationLayer(f64){
    .n_inputs = 32 * 26 * 26,
    .n_neurons = 32 * 26 * 26,
    .activationFunction = ActivationType.ReLU,
    .allocator = &allocator,
};
var conv1_act = ActivationLayer(f64).create(&conv1_activ);
try conv1_act.init(&allocator, @constCast(&conv1_activ));
try model.addLayer(conv1_act);
```

#### **Fully Connected (Dense) Layer**

```zig
var dense1 = DenseLayer(f64){
    .n_inputs = 64 * 5 * 5,
    .n_neurons = 512,
    .allocator = &allocator,
};
var dense1_layer = DenseLayer(f64).create(&dense1);
try dense1_layer.init(&allocator, @constCast(&dense1));
try model.addLayer(dense1_layer);
```

### **3. Load Data and Train the Model**

```zig
var load = loader.DataLoader(f64, u8, u8, 64, 3){};
const image_file_name: []const u8 = "datasets/t10k-images-idx3-ubyte";
const label_file_name: []const u8 = "datasets/t10k-labels-idx1-ubyte";
try load.loadMNIST2DDataParallel(&allocator, image_file_name, label_file_name);

try Trainer.TrainDataLoader2D(
    f64, u8, u8, &allocator, 64, 784, &model, &load, 30,
    LossType.CCE, 0.005, 0.9, 0.0001, 1.0
);

model.deinit();
```

---

## Docker

Follow the [Docker Guide](/docs/How_TO_DOCKER_101.md) for containerized usage.

## Join Us!

Contribute to Zant on [GitHub](#). Let’s make NN deployment on microcontrollers efficient, accessible, and open!

## Contributors

[View all contributors](https://github.com/ZIGTinyBook/Z-Ant/graphs/contributors)

