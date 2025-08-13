# Z-Ant

<div align="left">
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-tests.yml/badge.svg" alt="Zig Tests" />
  <img src="https://github.com/ZantFoundation/Z-Ant/actions/workflows/zant-benchmarks.yml/badge.svg" alt="Zig Benchamrk Tests" />
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-codegen-tests.yml/badge.svg" alt="Zig Codegen Tests" />
</div>

![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)


## Project Overview

## ✨ Why Z-Ant?

- **⚡ Microsecond** inference on ARM Cortex-M, RISC-V, x86
- **📦 Zero dependencies** - single static library deployment
- **🎯 ONNX native** - direct model deployment from ONNX
- **🔧 30+ operators** - comprehensive neural network support
- **📷 Built-in image processing** - JPEG decode + preprocessing
- **🧠 Smart optimization** - quantization, pruning, memory efficiency

## Use Cases

- **🏭 Edge AI**: Real-time anomaly detection, predictive maintenance
- **🤖 IoT & Autonomous Systems**: Lightweight AI models for drones, robots, vehicles, IoT devices
- **📱 Mobile Applications**: On-device inference for privacy-preserving AI
- **🏥 Medical Devices**: Real-time health monitoring and diagnostics
- **🎮 Gaming**: AI-powered gameplay enhancement on embedded systems

---

## 🚀 Quick Start
Prerequisites

- [Zig 0.14.1+](https://ziglang.org/learn/getting-started/)

### Get Started in 2 Minutes
```bash
# Clone and verify installation
git clone https://github.com/ZantFoundation/Z-Ant.git
cd Z-Ant
zig build test --summary all

# Generate code from your ONNX model
zig build codegen -Dmodel=mnist-1

# Build optimized library for ARM Cortex-M33
zig build lib -Dmodel=mnist-1 -Dtarget=thumb-freestanding -Dcpu=cortex_m33
```

## 📖 Essential Commands  

**IMPORTANT**: see [ZANT CLI](docs/ZANT_CLI.md) for a better understanding and more details!

### Core Workflow
| Command | What it does |
|---------|--------------|
| `zig build test` | Verify everything works |
| `zig build codegen -Dmodel=<name>` | Generate code from ONNX model |
| `zig build lib -Dmodel=<name>` | Build deployable static library |
| `zig build test-generated-lib -Dmodel=<name>` | Test your generated code |

### Target Platforms
| Platform | Target Flag | CPU Examples |
|----------|-------------|--------------|
| **ARM Cortex-M** | `-Dtarget=thumb-freestanding` | `-Dcpu=cortex_m33`, `-Dcpu=cortex_m4` |
| **RISC-V** | `-Dtarget=riscv32-freestanding` | `-Dcpu=generic_rv32` |
| **x86/Native** | `-Dtarget=native` | (auto-detected) |

### Key Options
| Option | Description | Example |
|--------|-------------|---------|
| `-Dmodel=<name>` | Your model name | `-Dmodel=my_classifier` |
| `-Dmodel_path=<path>` | Custom ONNX file | `-Dmodel_path=models/custom.onnx` |
| `-Dlog=true` | Enable detailed logging | `-Dlog=true` |
| `-Dcomm=true` | Add comments to generated code | `-Dcomm=true` |

## 🔧 ONNX Tools (Python Helpers)

Z-Ant includes Python scripts for ONNX model preparation:

```bash
# Prepare your model: set input shapes and infer all tensor shapes
./zant input_setter --path model.onnx --shape 1,3,224,224

# Generate test data for validation
./zant user_tests_gen --model model.onnx --iterations 10

# Create operator test models
./zant onnx_gen --op Conv --iterations 5
```

## 💼 Integration Examples

### CMake Integration
```cmake
target_link_libraries(your_project PUBLIC path/to/libzant.a)
```

### Arduino/Embedded C
```c
#include "lib_my_model.h"

// Optional: Set custom logging
extern void setLogFunction(void (*log_function)(uint8_t *string));

// Your inference code here
```

## 🎯 Real-World Examples

### Image Classification on Cortex-M33
```bash
# Generate optimized library for image classifier
zig build codegen -Dmodel=mobilenet_v2 -Dmodel_path=models/mobilenet.onnx
zig build lib -Dmodel=mobilenet_v2 -Dtarget=thumb-freestanding -Dcpu=cortex_m33 -Doutput_path=deployment/
```

### Multi-Platform Testing
```bash
# Test on different architectures
zig build test-generated-lib -Dmodel=my_model -Dtarget=native
zig build test-generated-lib -Dmodel=my_model -Dtarget=thumb-freestanding -Dcpu=cortex_m4
```

## 🛠️ Development

### For Contributors
```bash
# Run full test suite
zig build test --summary all

# Test heavy computational operations  
zig build test -Dheavy=true

# Test specific operator implementations
zig build op-codegen-test -Dop=Conv

# Generate and test single operations
zig build op-codegen-gen -Dop=Add
```

### Project Structure

```
Z-Ant/
├── src/                    # Core source code
│   ├── Core/              # Neural network core functionality
│   ├── CodeGen/           # Code generation engine
│   ├── ImageToTensor/     # Image preprocessing pipeline
│   ├── onnx/              # ONNX model parsing
│   └── Utils/             # Utilities and helpers
├── tests/                 # Comprehensive test suite
├── datasets/              # Sample models and test data
├── generated/             # Generated code output
├── examples/              # Arduino and microcontroller examples
└── docs/                  # Documentation and guides
```

---

## 🤝 Contributing

We welcome contributions from developers of all skill levels! Here's how to get involved:

### Getting Started
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your work
4. **Make your changes** following our coding standards
5. **Run tests** to ensure everything works
6. **Submit a pull request** for review

### Ways to Contribute
- **🐛 Bug Reports**: Found an issue? Let us know!
- **✨ Feature Requests**: Have an idea? Share it with us!
- **💻 Code Contributions**: Improve the codebase or add new features
- **📚 Documentation**: Help make the project easier to understand
- **🧪 Testing**: Write tests or improve test coverage

### Community Guidelines
- Follow our [Code of Conduct](docs/CODE_OF_CONDUCT.md)
- Check out the [Contributing Guide](docs/CONTRIBUTING.md) for detailed guidelines
- Join discussions on GitHub Issues and Discussions

### Recognition
All contributors are recognized in our [Contributors list](https://github.com/ZIGTinyBook/Z-Ant/contributors). Thank you for helping shape the future of tinyML!

---

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file in the repository.

---

<div align="center">

**Join us in revolutionizing AI on edge devices! 🚀**

[GitHub](https://github.com/ZIGTinyBook/Z-Ant) • [Documentation](docs/) • [Examples](examples/) • [Community](https://github.com/ZIGTinyBook/Z-Ant/discussions)

</div>