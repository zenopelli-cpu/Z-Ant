#ifndef ZANTAI_H
#define ZANTAI_H

#include <Arduino.h>
#include <stdint.h>
#include <stddef.h>

// Forward declaration for callback type
typedef int (*zant_weight_read_callback_t)(size_t offset, uint8_t* buffer, size_t size);

// External C functions from Zant library
extern "C" {
    int predict(float* input, uint32_t* input_shape, uint32_t shape_len, float** result);
    void zant_register_weight_callback(zant_weight_read_callback_t callback);
    void zant_init_weights_io(void);
}

class ZantAI {
public:
    // Initialize the Zant AI system
    void begin();
    
    // Set custom weight reading callback
    void setWeightCallback(zant_weight_read_callback_t callback);
    
    // Predict from RGB image (32x32x3)
    int predictImage(uint8_t* rgb_image, int width, int height, float* output);
    
    // Predict from raw float data (3072 elements)
    int predictRaw(float* input_data, float* output);
    
    // Print prediction results with class names
    void printPrediction(float* probabilities, const char** class_names, int num_classes);
    
    // Check if system is initialized
    bool isInitialized();
    
private:
    // Internal state tracking
};

#endif // ZANTAI_H 