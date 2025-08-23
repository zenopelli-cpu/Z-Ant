#include "ZantAI.h"
#include <Arduino.h>

// Global variables
static zant_weight_read_callback_t current_callback = nullptr;
static float input_buffer[3072]; // 1*3*32*32 for MobileNet input
static uint32_t input_shape[4] = {1, 3, 32, 32};
static float* prediction_result = nullptr;

// Flash storage callback for Nicla Vision
int flash_weight_callback(size_t offset, uint8_t* buffer, size_t size) {
    // Nicla Vision flash reading implementation
    // Assuming weights are stored in a dedicated flash section
    extern uint8_t _weights_start; // Linker symbol for weights section
    
    uint8_t* flash_ptr = &_weights_start + offset;
    memcpy(buffer, flash_ptr, size);
    return 0; // Success
}

void ZantAI::begin() {
    Serial.println("Initializing Zant AI...");
    
    // Initialize weights I/O system
    zant_init_weights_io();
    
    // Set up flash reading callback
    zant_register_weight_callback(flash_weight_callback);
    current_callback = flash_weight_callback;
    
    Serial.println("Zant AI initialized successfully!");
}

void ZantAI::setWeightCallback(zant_weight_read_callback_t callback) {
    zant_register_weight_callback(callback);
    current_callback = callback;
}

int ZantAI::predictImage(uint8_t* rgb_image, int width, int height, float* output) {
    if (!rgb_image || !output) {
        Serial.println("Error: null pointers provided");
        return -1;
    }
    
    // Resize and normalize image to 32x32x3 format
    if (width != 32 || height != 32) {
        Serial.println("Error: image must be 32x32 pixels");
        return -2;
    }
    
    // Convert RGB888 to normalized float array
    // Layout: [R0,R1,...,R1023, G0,G1,...,G1023, B0,B1,...,B1023]
    for (int i = 0; i < 1024; i++) {
        input_buffer[i] = (float)rgb_image[i*3] / 255.0f;        // R channel
        input_buffer[i + 1024] = (float)rgb_image[i*3 + 1] / 255.0f; // G channel  
        input_buffer[i + 2048] = (float)rgb_image[i*3 + 2] / 255.0f; // B channel
    }
    
    // Call Zant prediction
    int result = predict(input_buffer, input_shape, 4, &prediction_result);
    
    if (result == 0 && prediction_result) {
        // Copy results (assuming 4 classes output)
        for (int i = 0; i < 4; i++) {
            output[i] = prediction_result[i];
        }
    }
    
    return result;
}

int ZantAI::predictRaw(float* input_data, float* output) {
    if (!input_data || !output) {
        return -1;
    }
    
    // Direct prediction call
    int result = predict(input_data, input_shape, 4, &prediction_result);
    
    if (result == 0 && prediction_result) {
        // Copy results
        for (int i = 0; i < 4; i++) {
            output[i] = prediction_result[i];
        }
    }
    
    return result;
}

void ZantAI::printPrediction(float* probabilities, const char** class_names, int num_classes) {
    Serial.println("Prediction results:");
    for (int i = 0; i < num_classes; i++) {
        Serial.print("  ");
        Serial.print(class_names[i]);
        Serial.print(": ");
        Serial.print(probabilities[i] * 100.0f, 2);
        Serial.println("%");
    }
}

bool ZantAI::isInitialized() {
    return current_callback != nullptr;
} 