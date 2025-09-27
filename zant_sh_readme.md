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
- **create** - runs all the below, to use just once all the below are tested
- **onnx_gen** - Generate fuzzed ONNX models and save execution data in JSON
- **onnx_extract** - For a N nodes model it creates N onnx models, one for each node with respective tests.  
- **user_tests_gen** - Run ONNX model multiple times with random inputs and save execution data  
- **shape_thief** - Upgrade your model with all intermediate tensor's shapes
- **input_setter** - Set input shape and infer shapes of the ONNX model

### Getting Help

```bash
# General help
./zant --help

# Script-specific help
./zant <script> --help
```

## Examples

See [zant_CLI](docs/ZANT_CLI.md) for more details!


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

### shape_thief
**Location:** `src/onnx/shape_thief.py`

### input_setter
**Location:** `src/onnx/input_setter.py`

**Flags:**
- `--model` - Name of your model (required)
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
    ├── shape_thief.py
    └── input_setter.py
```