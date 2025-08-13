# Zant CLI Commands Reference

Zant is a tensor computation framework with ONNX support. This document provides a comprehensive reference for all available CLI commands and options.

## Build System Commands (Zig)

All Zig build commands follow the pattern: `zig build <command> [options]`

### Core Build Options

These options can be used with most build commands:

- `--target <string>` - Target architecture (e.g., `thumb-freestanding`, default: `native`)
- `--cpu <string>` - CPU model (e.g., `cortex_m33`)
- `-Dtrace_allocator=<bool>` - Use a tracing allocator (default: `true`)
- `-Dallocator=<string>` - Allocator to use (default: `raw_c_allocator`)

### Testing Commands

#### `zig build test`
Run all unit tests

**Options:**
- `-Dheavy=<bool>` - Run heavy tests (default: `false`)
- `-Dtest_name=<string>` - Specify a test name to run (default: `""`)

#### `zig build test-generated-lib`
Run generated library tests

#### `zig build op-codegen-test`
Run one-operation code generation tests

**Options:**
- `-Dop=<string>` - Operator name to test (default: `all`)

#### `zig build extractor-test`
Start extracted nodes tests

**Options:**
- `-Dmodel=<string>` - Model name (default: `mnist-8`)

#### `zig build onnx-parser`
Run ONNX parser tests

### Code Generation Commands

#### `zig build codegen`
Generate code from ONNX models

**Model Options:**
- `-Dmodel=<string>` - Model name (default: `mnist-8`)
- `-Dmodel_path=<string>` - Model path (default: `datasets/models/{model}/{model}.onnx`)
- `-Dgenerated_path=<string>` - Generated code output path (default: `generated/{model}/`)

**Code Generation Options:**
- `-Denable_user_tests=<bool>` - Enable user tests (default: `false`)
- `-Dlog=<bool>` - Run with logging (default: `false`)
- `-Dshape=<string>` - Input shape (default: `""`)
- `-Dtype=<string>` - Input type (default: `f32`)
- `-Doutput_type=<string>` - Output type (default: `f32`)
- `-Dcomm=<bool>` - Generate with comments (default: `false`)
- `-Ddynamic=<bool>` - Dynamic allocation (default: `false`)
- `-Ddo_export=<bool>` - Generate exportable code (default: `false`)
- `-Dv=<string>` - Version, v1 or v2 (default: `v1`)

#### `zig build op-codegen-gen`
Generate code for one-operation models

**Options:**
- `-Dop=<string>` - Operator name (default: `all`)

#### `zig build extractor-gen`
Generate tests for extracted nodes

**Options:**
- `-Dmodel=<string>` - Model name (default: `mnist-8`)

### Library Building Commands

#### `zig build lib`
Compile tensor_math static library

**Options:**
- `-Dmodel=<string>` - Model name (default: `mnist-8`)
- `-Doutput_path=<string>` - Output path for the library (default: `""`)

#### `zig build build-main`
Build the main executable for profiling

### Benchmark Commands

#### `zig build benchmark`
Run benchmarks

**Options:**
- `-Dfull=<bool>` - Run full benchmark (default: `false`)

## Python Tools Wrapper (Shell Script)

The `./zant` shell script provides a convenient wrapper for Python ONNX tools.

### General Usage
```bash
./zant <script> [flags]
./zant <script> --help  # Show script-specific help
```

### Available Python Scripts

#### `onnx_gen`
Generate fuzzed ONNX models and save execution data in JSON

**Usage:**
```bash
./zant onnx_gen [options]
```

**Options:**
- `--iterations <number>` - Number of models to generate for each operation (default: 1)
- `--seed <number>` - Seed for random generation (for reproducibility)
- `--output-dir <path>` - Directory to save generated models (default: `datasets/oneOpModels`)
- `--metadata-file <path>` - File to save metadata and execution data (default: `datasets/oneOpModels/results.json`)
- `--op <string>` - Name of the operation to generate and test (default: `all`)

**Example:**
```bash
./zant onnx_gen --iterations 5 --seed 42 --op Add
```

#### `user_tests_gen`
Run ONNX model multiple times with random inputs and save execution data

**Usage:**
```bash
./zant user_tests_gen --model <path> [options]
```

**Options:**
- `--model <path>` - Your ONNX model (required)
- `--iterations <number>` - Number of randomized inference runs (default: 1)

**Example:**
```bash
./zant user_tests_gen --model my_model.onnx --iterations 10
```

#### `infer_shape`
Upgrade your model with all intermediate tensor shapes

**Usage:**
```bash
./zant infer_shape --path <path>
```

**Options:**
- `--path <path>` - Path of your model (required)

**Example:**
```bash
./zant infer_shape --path model.onnx
```

#### `input_setter`
Set input shape and infer shapes of the ONNX model

**Usage:**
```bash
./zant input_setter --path <path> --shape <shape>
```

**Options:**
- `--path <path>` - Path of your model (required)
- `--shape <shape>` - Input shape as comma-separated values, e.g., `1,3,224,224` (required)

**Example:**
```bash
./zant input_setter --path model.onnx --shape 1,3,224,224
```

## Common Workflows

### 1. Generate and Test a Model
```bash
# Generate code for a specific model
zig build codegen -Dmodel=my_model -Dmodel_path=path/to/model.onnx

# Test the generated code
zig build test-generated-lib -Dmodel=my_model

# Build the static library
zig build lib -Dmodel=my_model
```

### 2. Test Single Operations
```bash
# Generate ONNX models for testing
./zant onnx_gen --op Add --iterations 5

# Generate code for one-op models
zig build op-codegen-gen -Dop=Add

# Test the generated operations
zig build op-codegen-test -Dop=Add
```

### 3. Prepare ONNX Models
```bash
# Set input shape and infer intermediate shapes
./zant input_setter --path model.onnx --shape 1,3,224,224

# Generate additional shape information
./zant infer_shape --path model.onnx

# Generate test data
./zant user_tests_gen --model model.onnx --iterations 10
```

### 4. Development and Debugging
```bash
# Run tests with detailed output
zig build test -Dheavy=true -Dlog=true

# Generate code with comments for debugging
zig build codegen -Dmodel=debug_model -Dcomm=true -Dlog=true

# Run benchmarks to check performance
zig build benchmark -Dfull=true
```

## Notes

- Most commands support the `--help` flag for detailed usage information
- The Zig build system uses `-D` prefix for custom options
- The shell wrapper automatically detects and uses `python3` or `python`
- Generated files are typically placed in the `generated/` directory
- Model files should be in ONNX format and placed in appropriate directories