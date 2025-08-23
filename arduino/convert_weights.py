#!/usr/bin/env python3
"""
Converter per pesi da Zig a C array per Arduino
Converte i pesi dal file static_parameters.zig in formato C utilizzabile su Arduino
"""

import re
import sys
import os
from pathlib import Path

def extract_arrays_from_zig(zig_file_path):
    """Estrae tutti gli array di pesi dal file Zig"""
    with open(zig_file_path, 'r') as f:
        content = f.read()
    
    # Pattern per trovare array di pesi
    array_pattern = r'const\s+array_([^:]+)\s*:\s*\[[^\]]+\]\s*([^=]+)\s*=\s*\[_\][^{]*\{([^}]+)\}'
    
    arrays = {}
    matches = re.finditer(array_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        array_name = match.group(1).strip()
        array_type = match.group(2).strip()
        array_data = match.group(3).strip()
        
        # Pulisci i dati dell'array
        values = []
        for line in array_data.split('\n'):
            line = line.strip()
            if line and not line.startswith('//'):
                # Rimuovi commenti inline
                line = re.sub(r'//.*$', '', line)
                # Estrai numeri (float o int)
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                values.extend(numbers)
        
        arrays[array_name] = {
            'type': array_type,
            'values': values,
            'count': len(values)
        }
        
    return arrays

def generate_c_header(arrays, output_file):
    """Genera file header C con gli array"""
    with open(output_file, 'w') as f:
        f.write('/*\n')
        f.write(' * Zant Model Weights - Generated from Zig\n')
        f.write(' * DO NOT EDIT - Auto-generated file\n')
        f.write(' */\n\n')
        f.write('#ifndef ZANT_WEIGHTS_H\n')
        f.write('#define ZANT_WEIGHTS_H\n\n')
        f.write('#include <stdint.h>\n')
        f.write('#include <stddef.h>\n\n')
        f.write('#ifdef __cplusplus\n')
        f.write('extern "C" {\n')
        f.write('#endif\n\n')
        
        # Dichiarazioni
        for name, array_info in arrays.items():
            c_type = 'float' if 'f32' in array_info['type'] else 'int32_t'
            f.write(f'extern const {c_type} array_{name}[{array_info["count"]}];\n')
        
        f.write('\n')
        f.write('#ifdef __cplusplus\n')
        f.write('}\n')
        f.write('#endif\n\n')
        f.write('#endif // ZANT_WEIGHTS_H\n')

def generate_c_source(arrays, output_file):
    """Genera file sorgente C con i dati degli array"""
    with open(output_file, 'w') as f:
        f.write('/*\n')
        f.write(' * Zant Model Weights Data - Generated from Zig\n')
        f.write(' * DO NOT EDIT - Auto-generated file\n')
        f.write(' */\n\n')
        f.write('#include "zant_weights.h"\n\n')
        
        for name, array_info in arrays.items():
            c_type = 'float' if 'f32' in array_info['type'] else 'int32_t'
            f.write(f'const {c_type} array_{name}[{array_info["count"]}] = {{\n')
            
            # Scrivi i valori, 8 per riga
            values = array_info['values']
            for i in range(0, len(values), 8):
                row = values[i:i+8]
                if c_type == 'float':
                    formatted_values = [f'{float(v):.6f}f' for v in row]
                else:
                    formatted_values = [str(int(float(v))) for v in row]
                
                f.write('    ' + ', '.join(formatted_values))
                if i + 8 < len(values):
                    f.write(',')
                f.write('\n')
            
            f.write('};\n\n')

def generate_weight_reader(arrays, output_file):
    """Genera funzione C per leggere i pesi tramite callback"""
    with open(output_file, 'w') as f:
        f.write('/*\n')
        f.write(' * Zant Weights Reader - Arduino Implementation\n')
        f.write(' * Provides weight reading functionality for embedded systems\n')
        f.write(' */\n\n')
        f.write('#include "zant_weights.h"\n')
        f.write('#include "ZantAI.h"\n\n')
        
        f.write('// Weight metadata table\n')
        f.write('typedef struct {\n')
        f.write('    const char* name;\n')
        f.write('    const void* data;\n')
        f.write('    size_t size;\n')
        f.write('    size_t offset;\n')
        f.write('} weight_entry_t;\n\n')
        
        f.write('static const weight_entry_t weight_table[] = {\n')
        offset = 0
        for name, array_info in arrays.items():
            element_size = 4  # float32 = 4 bytes
            total_size = array_info['count'] * element_size
            f.write(f'    {{"{name}", array_{name}, {total_size}, {offset}}},\n')
            offset += total_size
        f.write('};\n\n')
        
        f.write('static const size_t weight_table_size = sizeof(weight_table) / sizeof(weight_entry_t);\n\n')
        
        f.write('// Arduino weight callback implementation\n')
        f.write('int arduino_weight_callback(size_t offset, uint8_t* buffer, size_t size) {\n')
        f.write('    // Find the correct weight entry\n')
        f.write('    for (size_t i = 0; i < weight_table_size; i++) {\n')
        f.write('        const weight_entry_t* entry = &weight_table[i];\n')
        f.write('        if (offset >= entry->offset && \n')
        f.write('            offset + size <= entry->offset + entry->size) {\n')
        f.write('            // Copy data from the correct array\n')
        f.write('            size_t local_offset = offset - entry->offset;\n')
        f.write('            const uint8_t* src = (const uint8_t*)entry->data + local_offset;\n')
        f.write('            memcpy(buffer, src, size);\n')
        f.write('            return 0; // Success\n')
        f.write('        }\n')
        f.write('    }\n')
        f.write('    return -1; // Error: offset not found\n')
        f.write('}\n\n')
        
        f.write('// Initialize Arduino weight system\n')
        f.write('void zant_arduino_init_weights() {\n')
        f.write('    zant_register_weight_callback(arduino_weight_callback);\n')
        f.write('}\n')

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_weights.py <static_parameters.zig>")
        sys.exit(1)
    
    zig_file = sys.argv[1]
    if not os.path.exists(zig_file):
        print(f"Error: {zig_file} not found")
        sys.exit(1)
    
    print(f"Converting weights from {zig_file}...")
    
    # Estrai array dal file Zig
    arrays = extract_arrays_from_zig(zig_file)
    
    if not arrays:
        print("No arrays found in the Zig file")
        sys.exit(1)
    
    print(f"Found {len(arrays)} weight arrays:")
    total_params = 0
    for name, info in arrays.items():
        print(f"  - {name}: {info['count']} elements ({info['type']})")
        total_params += info['count']
    
    print(f"Total parameters: {total_params:,}")
    
    # Genera file di output
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    header_file = os.path.join(base_dir, "zant_weights.h")
    source_file = os.path.join(base_dir, "zant_weights.cpp")
    reader_file = os.path.join(base_dir, "zant_weight_reader.cpp")
    
    print(f"Generating {header_file}...")
    generate_c_header(arrays, header_file)
    
    print(f"Generating {source_file}...")
    generate_c_source(arrays, source_file)
    
    print(f"Generating {reader_file}...")
    generate_weight_reader(arrays, reader_file)
    
    print("Conversion complete!")
    print("\nNext steps:")
    print("1. Copy zant_weights.h, zant_weights.cpp, and zant_weight_reader.cpp to your Arduino sketch")
    print("2. Include them in your sketch compilation")
    print("3. Call zant_arduino_init_weights() in your setup() function")

if __name__ == "__main__":
    main() 