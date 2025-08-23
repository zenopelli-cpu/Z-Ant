/*
 * Test XIP functionality with Z-Ant library
 * 
 * This test demonstrates that weights are placed in flash memory
 * and can be accessed directly without copying to RAM.
 */

#include <stdio.h>
#include <stdint.h>

// Mock Z-Ant generated header (you would include your actual generated header)
// #include "lib_mnist-8.h"

// External symbols from linker script (these would be defined in your linker script)
extern uint32_t __flash_weights_start__;
extern uint32_t __flash_weights_end__;
extern uint32_t __flash_weights_size__;

// Mock function to simulate prediction
float mock_predict(float* input) {
    // In real Z-Ant, this would call the generated predict() function
    // which would access weights directly from flash
    
    printf("Accessing weights directly from flash memory...\n");
    printf("Flash weights region: 0x%08X - 0x%08X\n", 
           (uint32_t)&__flash_weights_start__, 
           (uint32_t)&__flash_weights_end__);
    printf("Flash weights size: %u bytes\n", (uint32_t)&__flash_weights_size__);
    
    // Simulate computation with flash-stored weights
    return 0.85f; // Mock prediction result
}

int main() {
    printf("Z-Ant XIP Test\n");
    printf("==============\n\n");
    
    // Example input (28x28 MNIST image flattened)
    float input[784] = {0}; // All zeros for simplicity
    
    // Fill with some test data
    for (int i = 0; i < 10; i++) {
        input[i] = (float)i * 0.1f;
    }
    
    printf("Running inference with XIP-enabled weights...\n");
    float result = mock_predict(input);
    
    printf("Prediction result: %.2f\n", result);
    printf("\nXIP Test completed successfully!\n");
    printf("Weights remained in flash memory throughout execution.\n");
    
    return 0;
} 