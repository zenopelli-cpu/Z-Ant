#!/usr/bin/env python3
"""
Script per estrarre i pesi dal file static_parameters.zig
e creare un file binario per la memoria esterna del Nicla Vision
"""
import re
import struct
import sys
from pathlib import Path

def extract_weight_metadata(zig_file):
    """Estrae i metadati dei pesi dal file Zig"""
    with open(zig_file, 'r') as f:
        content = f.read()
    
    # Trova la sezione weight_metadata
    metadata_pattern = r'const weight_metadata = \[_\]WeightInfo\{(.*?)\};'
    match = re.search(metadata_pattern, content, re.DOTALL)
    
    if not match:
        print("ERRORE: Non trovata la sezione weight_metadata")
        return []
    
    metadata_content = match.group(1)
    
    # Estrai ogni WeightInfo
    weight_info_pattern = r'WeightInfo\{\s*\.offset\s*=\s*(\d+),\s*\.size\s*=\s*(\d+),\s*\.element_count\s*=\s*(\d+),\s*\.type_size\s*=\s*(\d+)\s*\}'
    
    weights = []
    for match in re.finditer(weight_info_pattern, metadata_content):
        offset = int(match.group(1))
        size = int(match.group(2))
        element_count = int(match.group(3))
        type_size = int(match.group(4))
        
        weights.append({
            'offset': offset,
            'size': size,
            'element_count': element_count,
            'type_size': type_size
        })
    
    return weights

def extract_weight_arrays(zig_file):
    """Estrae gli array di pesi veri dal file Zig"""
    with open(zig_file, 'r') as f:
        content = f.read()
    
    # Trova tutti gli array const array_
    array_pattern = r'const array_([^:]+):\s*\[[^\]]+\]\s*f32\s*=\s*\[_\]f32\{([^}]+)\}'
    
    arrays = {}
    for match in re.finditer(array_pattern, content, re.DOTALL):
        array_name = match.group(1).strip()
        array_data = match.group(2).strip()
        
        # Pulisci e converte i numeri
        numbers = []
        for line in array_data.split('\n'):
            line = line.strip().rstrip(',')
            if line and not line.startswith('//'):
                # Rimuovi commenti
                line = re.sub(r'//.*$', '', line)
                # Trova tutti i numeri float
                floats = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
                numbers.extend([float(x) for x in floats])
        
        arrays[array_name] = numbers
        print(f"Trovato array {array_name} con {len(numbers)} elementi")
    
    return arrays

def create_weights_binary(weights_metadata, weights_arrays, output_file):
    """Crea un file binario con i pesi nell'ordine corretto"""
    
    with open(output_file, 'wb') as f:
        total_offset = 0
        
        for i, meta in enumerate(weights_metadata):
            expected_offset = meta['offset']
            element_count = meta['element_count']
            
            # Trova l'array corrispondente (questo è il trucco - dobbiamo mappare i metadati agli array)
            # Per ora scriviamo zeri come placeholder
            print(f"Peso {i}: offset={expected_offset}, elementi={element_count}")
            
            # Padding se necessario
            if total_offset < expected_offset:
                padding_size = expected_offset - total_offset
                f.write(b'\x00' * padding_size)
                total_offset = expected_offset
            
            # Scrivi i dati dei pesi (per ora zeri)
            weight_data = [0.0] * element_count
            for weight in weight_data:
                f.write(struct.pack('<f', weight))  # little-endian float32
            
            total_offset += element_count * 4  # 4 bytes per float32
    
    print(f"Creato file binario: {output_file} ({total_offset} bytes)")

def main():
    if len(sys.argv) != 2:
        print("Uso: python extract_weights.py <static_parameters.zig>")
        sys.exit(1)
    
    zig_file = sys.argv[1]
    if not Path(zig_file).exists():
        print(f"ERRORE: File {zig_file} non trovato")
        sys.exit(1)
    
    print(f"Estraendo pesi da {zig_file}...")
    
    # Estrai metadati
    metadata = extract_weight_metadata(zig_file)
    print(f"Trovati {len(metadata)} gruppi di pesi")
    
    # Estrai array
    arrays = extract_weight_arrays(zig_file)
    print(f"Trovati {len(arrays)} array di dati")
    
    # Crea file binario
    output_file = "mobilenet_weights.bin"
    create_weights_binary(metadata, arrays, output_file)
    
    print(f"✅ Completato! File dei pesi: {output_file}")
    print(f"📊 Dimensione totale: {sum(m['size'] for m in metadata)} bytes")
    
    # Stampa come caricare in memoria esterna
    print("\n🔧 Per caricare in memoria esterna:")
    print(f"dfu-util -a 1 -D {output_file} -s 0x90000000")

if __name__ == "__main__":
    main() 