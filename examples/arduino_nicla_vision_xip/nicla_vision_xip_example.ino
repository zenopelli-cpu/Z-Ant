/*
 * Arduino Nicla Vision XIP Example
 * 
 * This example shows how to use Z-Ant AI inference library with XIP (eXecute In Place)
 * on the Arduino Nicla Vision board. With XIP, model weights remain in flash memory
 * and are accessed directly, saving precious RAM.
 * 
 * Hardware: Arduino Nicla Vision (STM32H747XI)
 * 
 * Before uploading:
 * 1. Generate your model with XIP support:
 *    zig build codegen -Dmodel=your_model -Dxip=true
 * 2. Build the static library:
 *    zig build lib -Dmodel=your_model -Dtarget=thumb-freestanding -Dcpu=cortex_m7 -Dxip=true
 * 3. Copy the generated libzant.a to your Arduino libraries folder
 * 4. Use the custom linker script for XIP support
 */

#include "lib_your_model.h"  // Replace with your actual model name
#include <Arduino.h>

// External symbol from linker script
extern uint32_t __flash_weights_start__;
extern uint32_t __flash_weights_end__;
extern uint32_t __flash_weights_size__;

// Custom logging function for Z-Ant
void zant_log_function(uint8_t *message) {
    Serial.print("Z-Ant: ");
    Serial.println((char*)message);
}

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(100);  // Wait for serial connection
    }
    
    Serial.println("Arduino Nicla Vision XIP Example");
    Serial.println("=================================");
    
    // Print memory information
    Serial.print("Flash weights start: 0x");
    Serial.println((uint32_t)&__flash_weights_start__, HEX);
    Serial.print("Flash weights end:   0x");
    Serial.println((uint32_t)&__flash_weights_end__, HEX);
    Serial.print("Flash weights size:  ");
    Serial.print((uint32_t)&__flash_weights_size__);
    Serial.println(" bytes");
    
    // Check available RAM
    Serial.print("Free RAM: ");
    Serial.print(freeRAM());
    Serial.println(" bytes");
    
    // Set up Z-Ant logging
    setLogFunction(zant_log_function);
    
    Serial.println("Z-Ant initialization complete!");
    Serial.println("Ready for inference...");
}

void loop() {
    // Example input data - replace with your actual input
    float input_data[784];  // Example for MNIST 28x28 image
    
    // Fill with example data (you would get this from camera/sensors)
    for (int i = 0; i < 784; i++) {
        input_data[i] = random(0, 255) / 255.0f;
    }
    
    Serial.println("\n--- Running Inference ---");
    
    // Measure inference time
    unsigned long start_time = micros();
    
    // Run prediction (replace with your actual predict function)
    // float* result = predict(input_data);
    
    unsigned long inference_time = micros() - start_time;
    
    Serial.print("Inference completed in: ");
    Serial.print(inference_time);
    Serial.println(" microseconds");
    
    // Print results (example)
    // Serial.print("Predicted class: ");
    // Serial.println(get_max_index(result, 10));  // For classification
    
    Serial.print("Free RAM after inference: ");
    Serial.print(freeRAM());
    Serial.println(" bytes");
    
    delay(2000);  // Wait before next inference
}

// Function to measure free RAM
int freeRAM() {
    char top;
    extern char *__brkval;
    extern char __bss_end;
    
    return __brkval ? &top - __brkval : &top - &__bss_end;
}

// Helper function to find max index (for classification tasks)
int get_max_index(float* array, int size) {
    int max_idx = 0;
    float max_val = array[0];
    
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
            max_idx = i;
        }
    }
    
    return max_idx;
} 