#!/bin/bash

#################################
# IMPORTANT!! Moove me int the root before ./profiling_script
# HOW TO USE:
# - add in models=() what you want to profile, ensure they are "well defined models", see docs/ZANT_CLI.md cahpter "1. MOST IMPORTANT - Generate and Test a Model "
# - run : chmod +x profiling_script.sh
# - run : ./profiling_script.sh
#
# It will create a /profiling folder where you can find everything ordered by model (static/dynamic)



# Array of models to profile
models=( "add_model" "gemm_model" "conv_model" "minimodel1" "qdq_model" "mnist-8") # "add_model" "gemm_model" "conv_model" "minimodel1" "qdq_model" "mnist-8"

# Create profiling directory if it doesn't exist
mkdir -p profiling

# Function to extract memory leak information from valgrind output
extract_memcheck_info() {
    local output_file="$1"
    echo "=== MEMORY LEAK ANALYSIS ===" >> "$output_file"
    
    # Extract leak summary
    grep -A 10 "LEAK SUMMARY:" valgrind_memcheck.log >> "$output_file" 2>/dev/null || echo "No leak summary found" >> "$output_file"
    
    # Extract heap summary
    grep -A 5 "HEAP SUMMARY:" valgrind_memcheck.log >> "$output_file" 2>/dev/null || echo "No heap summary found" >> "$output_file"
    
    # Extract error summary
    grep "ERROR SUMMARY:" valgrind_memcheck.log >> "$output_file" 2>/dev/null || echo "No error summary found" >> "$output_file"
    
    echo "" >> "$output_file"
}

# Function to extract callgrind information
extract_callgrind_info() {
    local output_file="$1"
    echo "=== CALLGRIND ANALYSIS ===" >> "$output_file"
    
    # Get callgrind output file (most recent)
    local callgrind_file=$(ls -t callgrind.out.* 2>/dev/null | head -n1)
    
    if [ -n "$callgrind_file" ]; then
        echo "Callgrind file: $callgrind_file" >> "$output_file"
        
        # Extract top functions with callgrind_annotate
        callgrind_annotate --threshold=1 "$callgrind_file" 2>/dev/null | head -n 30 >> "$output_file" || echo "Failed to analyze callgrind output" >> "$output_file"
        
        # Move callgrind file to profiling directory
        mv "$callgrind_file" "profiling/"
    else
        echo "No callgrind output file found" >> "$output_file"
    fi
    
    echo "" >> "$output_file"
}

# Function to extract massif information
extract_massif_info() {
    local output_file="$1"
    echo "=== MASSIF ANALYSIS ===" >> "$output_file"
    
    # Get massif output file (most recent)
    local massif_file=$(ls -t massif.out.* 2>/dev/null | head -n1)
    
    if [ -n "$massif_file" ]; then
        echo "Massif file: $massif_file" >> "$output_file"
        
        # Extract heap usage summary
        ms_print "$massif_file" 2>/dev/null | head -n 50 >> "$output_file" || echo "Failed to analyze massif output" >> "$output_file"
        
        # Move massif file to profiling directory
        mv "$massif_file" "profiling/"
    else
        echo "No massif output file found" >> "$output_file"
    fi
    
    echo "" >> "$output_file"
}

# Function to get executable size in a readable format
get_executable_size() {
    if [ -f "./zig-out/bin/main_profiling_target" ]; then
        ls -lh ./zig-out/bin/main_profiling_target | awk '{print $5}'
    else
        echo "N/A (executable not found)"
    fi
}

# Function to profile a model (static or dynamic)
profile_model() {
    local model="$1"
    local mode="$2" # "static" or "dynamic"
    local output_dir="profiling/${model}_${mode}"
    
    echo "Profiling $model ($mode mode)..."
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Output file for this configuration
    local report_file="$output_dir/profiling_report.txt"
    
    # Clear previous report
    > "$report_file"
    
    echo "=====================================" >> "$report_file"
    echo "PROFILING REPORT: $model ($mode)" >> "$report_file"
    echo "Date: $(date)" >> "$report_file"
    echo "=====================================" >> "$report_file"
    echo "" >> "$report_file"
    
    # Build commands based on mode
    if [ "$mode" == "static" ]; then
        echo "Building static version..." >> "$report_file"
        zig build lib-gen -Dmodel="$model" -Ddo_export 2>&1 | tee -a "$report_file"

        zig build build-main -Dmodel="$model" -Doptimize=ReleaseSmall 2>&1 | tee -a "$report_file"
    else
        echo "Building dynamic version..." >> "$report_file"
        zig build lib-gen -Dmodel="$model" -Ddo_export -Ddynamic 2>&1 | tee -a "$report_file"
        zig build build-main -Dmodel="$model" 2>&1 -Doptimize=ReleaseSmall | tee -a "$report_file"
    fi
    
    # Check if executable was created
    if [ ! -f "./zig-out/bin/main_profiling_target" ]; then
        echo "ERROR: Executable not found after build!" >> "$report_file"
        return 1
    fi
    
    # Get executable size
    local exe_size=$(get_executable_size)
    echo "Executable size: $exe_size" >> "$report_file"
    echo "" >> "$report_file"
    
    # Clean up old valgrind output files
    rm -f valgrind_*.log callgrind.out.* massif.out.*
    
    # Run Valgrind Memcheck
    echo "Running Valgrind Memcheck..." >> "$report_file"
    valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
             --log-file=valgrind_memcheck.log \
             ./zig-out/bin/main_profiling_target 2>&1
    
    extract_memcheck_info "$report_file"
    
    # Run Valgrind Callgrind
    echo "Running Valgrind Callgrind..." >> "$report_file"
    valgrind --tool=callgrind --callgrind-out-file=callgrind.out.%p \
             ./zig-out/bin/main_profiling_target > /dev/null 2>&1
    
    extract_callgrind_info "$report_file"
    
    # Run Valgrind Massif
    echo "Running Valgrind Massif..." >> "$report_file"
    valgrind --tool=massif --time-unit=ms --massif-out-file=massif.out.%p \
             ./zig-out/bin/main_profiling_target > /dev/null 2>&1
    
    extract_massif_info "$report_file"
    
    # Move valgrind log files to output directory
    mv valgrind_*.log "$output_dir/" 2>/dev/null || true
    
    echo "Profiling complete for $model ($mode). Results saved in $output_dir/"
    echo "========================================" >> "$report_file"
    echo "" >> "$report_file"
}

# Main execution
echo "Starting profiling for all models..."
echo "This may take a while as Valgrind significantly slows execution."

for model in "${models[@]}"; do
    echo ""
    echo "Processing model: $model"
    echo "=================================="
    
    # Profile static version
    profile_model "$model" "static"
    
    # Profile dynamic version  
    profile_model "$model" "dynamic"
done

echo ""
echo "All profiling complete!"
echo "Results are organized in the 'profiling/' directory:"
echo "- profiling/MODEL_NAME_static/"
echo "- profiling/MODEL_NAME_dynamic/"
echo ""
echo "Each directory contains:"
echo "  - profiling_report.txt (main report)"
echo "  - valgrind_memcheck.log (detailed memcheck output)"
echo "  - callgrind.out.* (callgrind data files)"
echo "  - massif.out.* (massif data files)"