ù#!/bin/bash

#################################
# IMPORTANT!! Move me into the root before ./profiling_script
# HOW TO USE:
# - add in models=() what you want to profile, ensure they are "well defined models", see docs/ZANT_CLI.md chapter "1. MOST IMPORTANT - Generate and Test a Model "
# - run : chmod +x profiling_script.sh
# - run : ./profiling_script.sh [OPTIONS]
#
# It will create a /profiling folder where you can find everything ordered by model with all optimization modes

# Default values
DEFAULT_MODELS=( "beer" "new2" "darknet_s" "fomo8" "mnist-8" )
DEFAULT_OPTIMIZE_MODES=( "ReleaseFast" "ReleaseSmall" )

# Initialize variables
models=()
optimize_modes=()
target=""
cpu=""

#zig build lib-gen \
#  -Dmodel="my_model" \
#  -Dxip=true \
#  -Ddynamic \
#  -Ddo_export \
#  -Denable_user_tests

#zig build lib \
#  -Dmodel="my_model"\
#  -Dtarget=thumb-freestanding \
#  -Dcpu=cortex_m7 \
#  -Dxip=true

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --models MODEL1,MODEL2,...     Comma-separated list of models to build"
    echo "                                      Default: beer,new2,darknet_s,fomo8,mnist-8"
    echo "  -o, --optimize MODE1,MODE2,...      Comma-separated list of optimization modes"
    echo "                                      Default: ReleaseFast,ReleaseSmall"
    echo "                                      Available: Debug,ReleaseSafe,ReleaseFast,ReleaseSmall"
    echo "  -t, --target TARGET                 Target architecture (e.g., x86_64-linux, aarch64-macos, native)"
    echo "                                      Optional - only added if specified"
    echo "  -c, --cpu CPU                       Target CPU features (e.g., baseline, native, x86_64, cortex_a72)"
    echo "                                      Optional - only added if specified"
    echo "  -h, --help                          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Use defaults, no target/cpu flags"
    echo "  $0 -m beer,mnist-8                         # Build only beer and mnist-8 models"
    echo "  $0 -o ReleaseFast                          # Only ReleaseFast optimization"
    echo "  $0 -t x86_64-linux -c x86_64               # With specific target and CPU"
    echo "  $0 -m beer -o ReleaseFast -t native        # Specific model, optimization and target"
    echo ""
    echo "Output structure:"
    echo "  profiling/"
    echo "  ├── modelName/"
    echo "  │   ├── ReleaseFast/"
    echo "  │   │   └── libzant.a"
    echo "  │   └── ReleaseSmall/"
    echo "  │       └── libzant.a"
    echo "  └── ..."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            IFS=',' read -ra models <<< "$2"
            shift 2
            ;;
        -o|--optimize)
            IFS=',' read -ra optimize_modes <<< "$2"
            shift 2
            ;;
        -t|--target)
            target="$2"
            shift 2
            ;;
        -c|--cpu)
            cpu="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Use defaults if not specified
if [ ${#models[@]} -eq 0 ]; then
    models=("${DEFAULT_MODELS[@]}")
fi

if [ ${#optimize_modes[@]} -eq 0 ]; then
    optimize_modes=("${DEFAULT_OPTIMIZE_MODES[@]}")
fi

echo "Configuration:"
echo "  Models to build: ${models[*]}"
echo "  Optimization modes: ${optimize_modes[*]}"
if [ -n "$target" ]; then
    echo "  Target: $target"
fi
if [ -n "$cpu" ]; then
    echo "  CPU: $cpu"
fi
echo ""

# Create profiling directory if it doesn't exist
mkdir -p profiling

# Function to create library with specified parameters
create_lib() {
    local model="$1"
    local optimize_mode="$2"
    local target="$3"
    local cpu="$4"
    
    echo "Creating $model.a with optimize=$optimize_mode"
    
    # Create target directory structure
    local target_dir="profiling/$model/$optimize_mode"
    mkdir -p "$target_dir"
    
    # Build the command dynamically
    local gen_build_cmd="zig build lib-gen -Dmodel=$model -Ddo_export -Dxip=true " 
    local lib_build_cmd="zig build lib -Dmodel=$model -Doptimize=$optimize_mode -Dxip=true " 
    
    # Only add target flag if specified
    if [ -n "$target" ]; then
        lib_build_cmd="$lib_build_cmd -Dtarget=$target"
        echo "  Target: $target"
    fi
    
    # Only add cpu flag if specified
    if [ -n "$cpu" ]; then
        lib_build_cmd="$lib_build_cmd -Dcpu=$cpu"
        echo "  CPU: $cpu"
    fi
    
    # Execute the build commands
    echo "  Command: $gen_build_cmd"
    if eval "$gen_build_cmd"; then

        echo "  Command: $lib_build_cmd"
        if eval "$lib_build_cmd"; then 
            # Check for the generated library file
            local source_lib=""
            
            # Look for the library in possible locations
            if [ -f "zig-out/lib/libzant.a" ]; then
                source_lib="zig-out/lib/libzant.a"
            elif [ -f "zig-out/$model/libzant.a" ]; then
                source_lib="zig-out/$model/libzant.a"
            elif [ -f "zig-out/lib/lib$model.a" ]; then
                source_lib="zig-out/lib/lib$model.a"
            else
                echo "  ✗ Error: Library file not found after build"
                echo "    Searched in:"
                echo "      - zig-out/lib/libzant.a"
                echo "      - zig-out/$model/libzant.a" 
                echo "      - zig-out/lib/lib$model.a"
                return 1
            fi
            
            # Move the library to the target directory
            if cp "$source_lib" "$target_dir/libzant.a"; then
                local lib_size=$(ls -lh "$target_dir/libzant.a" | awk '{print $5}')
                echo "  ✓ Success: Library moved to $target_dir/libzant.a (Size: $lib_size)"
            else
                echo "  ✗ Error: Failed to move library to $target_dir/"
                return 1
            fi
            
            # Clean up the zig-out directory for next build
            rm -rf zig-out
            
        else
            echo "  ✗ Error: Build failed for $model with lib.a creation"
            return 1
        fi
    else
        echo "  ✗ Error: Build failed for $model with: zig build lib-gen ..."
        return 1
    fi
    
    echo ""
    return 0
}

# Function to build a model with all optimization modes
build_model() {
    local model="$1"
    local target="$2"
    local cpu="$3"
    
    echo "Processing model: $model"
    echo "=================================="
    
    local success_count=0
    local total_count=${#optimize_modes[@]}
    
    # Build with each optimization mode
    for optimize_mode in "${optimize_modes[@]}"; do
        if create_lib "$model" "$optimize_mode" "$target" "$cpu"; then
            ((success_count++))
        fi
    done
    
    echo "Model $model completed: $success_count/$total_count successful builds"
    echo ""
    
    return 0
}

# Main execution
echo "Starting library generation for all models..."
echo "Optimization modes to build: ${optimize_modes[*]}"
echo ""

total_success=0
total_builds=0

for model in "${models[@]}"; do
    build_model "$model" "$target" "$cpu"
    
    # Count successful builds
    for optimize_mode in "${optimize_modes[@]}"; do
        ((total_builds++))
        if [ -f "profiling/$model/$optimize_mode/libzant.a" ]; then
            ((total_success++))
        fi
    done
done

echo "==============================================="
echo "BUILD SUMMARY"
echo "==============================================="
echo "Total successful builds: $total_success/$total_builds"
echo ""
echo "Generated libraries organized in:"
echo "profiling/"
for model in "${models[@]}"; do
    echo "├── $model/"
    for optimize_mode in "${optimize_modes[@]}"; do
        if [ -f "profiling/$model/$optimize_mode/libzant.a" ]; then
            local size=$(ls -lh "profiling/$model/$optimize_mode/libzant.a" | awk '{print $5}')
            echo "│   ├── $optimize_mode/ → libzant.a ($size)"
        else
            echo "│   ├── $optimize_mode/ → ✗ FAILED"
        fi
    done
done
echo ""
echo "All library generation complete!"