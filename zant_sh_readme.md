# Zant - ONNX Python Tools Wrapper

A shell script wrapper for various ONNX Python tools that provides a unified interface with proper flag handling and help documentation.

## Installation

1. Make the script executable:
```bash
chmod +x zant
```

2. Optionally, add it to your PATH or create a symlink:
```bash
# Option 1: Copy to a directory in your PATH
sudo cp zant /usr/local/bin/

# Option 2: Create a symlink
ln -s $(pwd)/zant /usr/local/bin/zant
```

## Usage

```bash
./zant <script> [flags]
```

### Available Scripts

- **onnx_gen** - Generate fuzzed ONNX models and save execution data in JSON
- **onnx_extract** - For a N nodes model it creates N onnx models, one for each node with respective tests.  
- **user_tests_gen** - Run ONNX model multiple times with random inputs and save execution data  
- **infer_shape** - Upgrade your model with all intermediate tensor's shapes
- **input_setter** - Set input shape and infer shapes of the ONNX model

### Getting Help

```bash
# General help
./zant --help

# Script-specific help
./zant <script> --help
```

## Examples

### ONNX Generation
```bash
# Generate 5 models with seed 42
./zant onnx_gen --iterations 5 --seed 42

# Generate models for specific operation
./zant onnx_gen --op Conv --iterations 3 --output-dir ./my_models

# Generate with all available flags
./zant onnx_gen --iterations 10 --seed 123 --output-dir ./output --metadata-file ./results.json --op Add
```

### ONNX Extraction
For a N nodes model it creates N onnx models, one for each node with respective tests.
```bash
./zant onnx_extract --path path/model.onnx 
```

### User Tests Generation
```bash
# Basic usage with required model flag
./zant user_tests_gen --model my_model

# Multiple iterations
./zant user_tests_gen --model my_model --iterations 10
```

### Shape Inference
```bash
# Infer shapes for a model
./zant infer_shape --path model.onnx
```

### Input Shape Setting
```bash
# Set input shape to 1x3x224x224
./zant input_setter --path path/model.onnx --shape 1,3,224,224

# Set input shape to batch size 4
./zant input_setter --path model.onnx --shape 4,3,256,256
```

## Script Details

### onnx_gen
**Location:** `tests/CodeGen/Python-ONNX/onnx_gen.py`

**Flags:**
- `--iterations` - Number of models to generate for each operation (default: 1)
- `--seed` - Seed for random generation (for reproducibility)
- `--output-dir` - Directory to save generated models (default: datasets/oneOpModels)
- `--metadata-file` - File to save metadata and execution data (default: datasets/oneOpModels/results.json)
- `--op` - Name of the operation you want to generate and test (default: all)

### user_tests_gen
**Location:** `tests/CodeGen/user_tests_gen.py`

**Flags:**   
- `--model` - Your ONNX model (required)
- `--iterations` - Number of randomized inference runs (default: 1)

### infer_shape
**Location:** `src/onnx/infer_shape.py`

**Flags:**
- `--path` - Path of your model (required)

### input_setter
**Location:** `src/onnx/input_setter.py`

**Flags:**
- `--path` - Path of your model (required)
- `--shape` - Input shape as comma-separated values, e.g., 1,3,34,34 (required)

## Requirements

- Python 3 (or Python 2 as fallback)
- The respective Python scripts must exist in their specified locations
- Required Python dependencies for each script (ONNX, numpy, etc.)

## Error Handling

The script includes comprehensive error handling:
- Validates script names
- Checks for Python script file existence
- Verifies Python interpreter availability
- Provides helpful error messages with usage instructions

## Directory Structure

The script expects the following directory structure:
```
project/
├── zant                                    # This wrapper script
├── tests/CodeGen/Python-ONNX/onnx_gen.py
├── tests/CodeGen/user_tests_gen.py
└── src/onnx/
    ├── infer_shape.py
    └── input_setter.py
```