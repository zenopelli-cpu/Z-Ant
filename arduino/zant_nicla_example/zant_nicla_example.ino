/*
 * Zant AI on Arduino Nicla Vision - Example
 * 
 * This example shows how to run MobileNet inference on Nicla Vision
 * using the Zant AI library with configurable weights I/O.
 */

#include "ZantAI.h"
#include "Arduino_BMI270_BMM150.h"  // For Nicla Vision sensors
#include "camera.h"                 // For Nicla Vision camera

// Create Zant AI instance
ZantAI zant;

// Model output classes (adjust for your model)
const char* class_names[] = {
    "Class 0",
    "Class 1", 
    "Class 2",
    "Class 3"
};
const int num_classes = 4;

// Image buffer for camera capture
uint8_t image_buffer[32 * 32 * 3]; // RGB 32x32
float predictions[4];               // Output probabilities

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(100);
    }
    
    Serial.println("Arduino Nicla Vision + Zant AI Demo");
    Serial.println("====================================");
    
    // Initialize camera
    if (!Camera.begin(CAMERA_R320x240, IMAGE_MODE_RGB888, 30)) {
        Serial.println("Failed to initialize camera!");
        while (1);
    }
    Serial.println("Camera initialized");
    
    // Initialize Zant AI
    zant.begin();
    
    if (!zant.isInitialized()) {
        Serial.println("Failed to initialize Zant AI!");
        while (1);
    }
    
    Serial.println("Setup complete! Taking pictures every 3 seconds...");
}

void loop() {
    // Capture image from camera
    Camera.grab();
    
    // Get pointer to camera buffer
    uint8_t* camera_data = Camera.getBuffer();
    
    if (camera_data) {
        // Resize/crop camera image to 32x32 (simplified - you'd want proper resizing)
        cropAndResize(camera_data, 320, 240, image_buffer, 32, 32);
        
        // Run AI inference
        int result = zant.predictImage(image_buffer, 32, 32, predictions);
        
        if (result == 0) {
            Serial.println("\\n--- Prediction Results ---");
            zant.printPrediction(predictions, class_names, num_classes);
            
            // Find highest probability class
            int best_class = 0;
            float best_prob = predictions[0];
            for (int i = 1; i < num_classes; i++) {
                if (predictions[i] > best_prob) {
                    best_prob = predictions[i];
                    best_class = i;
                }
            }
            
            Serial.print("Best prediction: ");
            Serial.print(class_names[best_class]);
            Serial.print(" (");
            Serial.print(best_prob * 100.0f, 1);
            Serial.println("%)");
            
        } else {
            Serial.print("Prediction failed with error: ");
            Serial.println(result);
        }
    } else {
        Serial.println("Failed to capture image");
    }
    
    delay(3000); // Wait 3 seconds between predictions
}

// Simple crop and resize function (you'd want a proper implementation)
void cropAndResize(uint8_t* src, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h) {
    // This is a very simplified version - just takes center crop and decimates
    int start_x = (src_w - dst_w) / 2;
    int start_y = (src_h - dst_h) / 2;
    
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int src_idx = ((start_y + y) * src_w + (start_x + x)) * 3;
            int dst_idx = (y * dst_w + x) * 3;
            
            dst[dst_idx] = src[src_idx];         // R
            dst[dst_idx + 1] = src[src_idx + 1]; // G  
            dst[dst_idx + 2] = src[src_idx + 2]; // B
        }
    }
} 