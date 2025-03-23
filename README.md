# Z-Ant

<div align="left">
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-tests.yml/badge.svg" alt="Zig Tests" />
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-heavy-tests.yml/badge.svg" alt="Zig Heavy Tests" />
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-codegen-tests.yml/badge.svg" alt="Zig Codegen Tests" />
</div>

![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)
## Project Overview


**Zant** (Zig-Ant) is an open-source SDK for deploying optimized neural networks (NNs) on microcontrollers.

## Why Zant?

- **Lack of DL Support**: Devices like TI Sitara, Raspberry Pi Pico, or ARM Cortex-M lack comprehensive DL libraries.
- **Open-source**: End-to-end NN deployment and optimization open-source solution.
- **Research-Inspired**: Implements optimization inspired by MIT's Han Lab research.
- **Academic Collaboration**: Developed in collaboration with institutions like Politecnico di Milano.

## Key Features

- **Real-time Optimizations**: Quantization, pruning, buffer optimization.
- **Cross-platform Compatibility**: ARM Cortex-M, RISC-V, and others.
- **Modular and Easy Integration**: Clear APIs, examples, and extensive documentation.

## Use Cases

- **Edge AI**: Real-time anomaly detection, predictive maintenance.
- **IoT & Autonomous Systems**: Lightweight AI models for drones, robots, vehicles, IoT devices.

---

## Roadmap

### Short-Term Goals (Q1 2025)

- **March 5**: MNIST inference on Raspberry Pi Pico 2.
- **April 30**: Efficient YOLO on Raspberry Pi Pico 2.

### Mid-Term Goals (Q2-Q3 2025)

- Shape Tracker implementation.
- Frontend GUI for library interaction.
- `im2tensor` for image preprocessing.
- Enhanced code generation (flash vs RAM execution).
- Expanded ONNX compatibility.

### Long-Term Goals (Q3 2025)

- Advanced pruning and quantization support.
- Expanded microcontroller compatibility.
- Model execution benchmarking tools.
- Improved real-time inference capabilities.

## Getting Started

### Requirements

- Install the latest [Zig compiler](https://ziglang.org/learn/getting-started/).
- Improve Zig proficiency via [Ziglings](https://codeberg.org/ziglings/exercises).

### Running the Project

```bash
zig build run
```

### Running Tests

Add tests to `build.zig/test_list`.

- Regular tests:
  ```bash
  zig build test --summary all
  ```
- Heavy computational tests:

```bash
zig build test -Dheavy --summary all
```

## Documentation

Follow [Zig's doc-comments](https://ziglang.org/documentation/master/#Doc-Comments).

## Using Zant

### Generating Code for Models

```bash
zig build codegen -Dmodel=model_name [-Dlog -Duser_tests=user_tests.json]
```

Generated code will be placed in:

```
generated/model_name/
├── lib_{model_name}.zig
├── test_{model_name}.zig
└── user_tests.json
```

### Testing Generated Models

```bash
zig build test-codegen -Dmodel=model_name
```

### Integrating into your Project

Build the static library:

```bash
zig build lib -Dmodel=model_name -Dtarget={arch} -Dcpu={cpu}
```

Linking with CMake:

```cmake
target_link_libraries(your_project PUBLIC path/to/libzant.a)
```

### Logging (Optional)

To set a custom log function from your C code:

```c
extern void setLogFunction(void (*log_function)(uint8_t *string));
```

## Zig Build File (`build.zig`)

**Key Build Commands:**

- **Standard build & test:**
  ```bash
  zig build
  zig build test
  ```

- **Run code generation:**
  ```bash
  zig build codegen -Dmodel=model_name [-Dlog -Duser_tests=path/to/tests.json]
  ```

- **Compile static library:**
  ```bash
  zig build lib -Dmodel=model_name -Dtarget=target_arch -Dcpu=specific_cpu
  ```

- **Run generated tests:**
  ```bash
  zig build test-codegen
  ```

### Build Options

- `-Dtrace_allocator=true|false`: Use tracing allocator for debugging.
- `-Dlog=true|false`: Enable detailed logging during code generation.
- `-Duser_tests=path/to/user_tests.json`: Specify custom tests.


##CI/CD Pipeline
- We are committed to enhancing our Continuous Integration/Continuous Deployment (CI/CD) pipeline to ensure robustness, reliability, and performance of Zant across all supported platforms. Key improvements include:
Hardware-in-the-Loop (HIL) Testing: Integrate a hardware test bench with connected microcontrollers (e.g., Raspberry Pi Pico, ARM Cortex-M) into the CI/CD pipeline to validate real-world performance and compatibility.

- Profiling in CI/CD: Automatically profile generated models during the pipeline to measure execution time, memory usage, and power consumption on target hardware.
- Daily Fuzzing Tests: Run fuzzing tests daily within the CI/CD pipeline to identify edge-case bugs and ensure model stability under unexpected inputs.

- Multi-Platform Build Matrix: Test builds across a variety of architectures (ARM, RISC-V) and configurations in parallel to catch platform-specific issues early.

- Automated Benchmarking: Include performance benchmarking in the pipeline to track inference speed and resource usage over time, ensuring optimizations don’t regress.

- Code Coverage Reporting: Generate and publish code coverage metrics with every CI run to maintain high test quality.
- Containerized CI Environment: Use Docker containers to standardize the CI/CD environment, ensuring consistent builds and tests across all contributors.




## Containerization

- Follow our [Docker guide](/docs/How_TO_DOCKER_101.md).

## Contributing

Join us on [GitHub](#) and shape the future of tinyML!

## Contributors

[All contributors](https://github.com/ZIGTinyBook/Z-Ant/contributors). Let's grow together!

