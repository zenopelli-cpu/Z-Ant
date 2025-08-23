#ifndef ZANT_ARDUINO_H
#define ZANT_ARDUINO_H

#include <Arduino.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for MobileNet model
typedef struct {
    float* data;
    size_t size;
} zant_tensor_t;

// Main prediction function - matches generated mobilenet_v2
extern int predict(
    float* input,           // Input image data (1, 3, 32, 32) = 3072 floats
    uint32_t* input_shape,  // Shape array: [1, 3, 32, 32]
    uint32_t shape_len,     // Length: 4
    float** result          // Output: pointer to result array
);

// Weights I/O functions
typedef int (*zant_weight_read_callback_t)(size_t offset, uint8_t* buffer, size_t size);

extern void zant_register_weight_callback(zant_weight_read_callback_t callback);
extern void zant_init_weights_io(void);

// Arduino convenience functions
void zant_arduino_init(void);
int zant_predict_image(uint8_t* rgb_image, int width, int height, float* output_probabilities);
void zant_set_flash_weights_callback(void);

#ifdef __cplusplus
}
#endif

#endif // ZANT_ARDUINO_H 