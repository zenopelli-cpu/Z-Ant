// UNCOMMENT THIS TO SEE A TEXTUAL REPRESENTATION OF THE PREPROCESSED IMAGE
// #define DEBUG_PREPROCESSED_IMAGE

#include <lib_zant.h> // Predict function header (expects const float* input)
#include "camera.h"
#define ARDUINO_NICLA_VISION

// --- CAMERA SELECTION (Nicla Vision) ---
#ifdef ARDUINO_NICLA_VISION
  #include "gc2145.h"
  GC2145 galaxyCore;
  Camera cam(galaxyCore);
  #define IMAGE_MODE 1
#else
#error "This board is not supported or not selected correctly in the IDE."
#endif
// --- END CAMERA SELECTION ---

// --- CONSTANTS and VARIABLES for PREDICTION ---
constexpr int MNIST_WIDTH = 28;
constexpr int MNIST_HEIGHT = 28;
constexpr int MNIST_CHANNELS = 1; // Grayscale
constexpr int MNIST_INPUT_SIZE = MNIST_WIDTH * MNIST_HEIGHT * MNIST_CHANNELS;
constexpr int MNIST_CLASSES = 10; // Output 0 to 9

// Buffer for the preprocessed image from camera (float, values 0.0f-255.0f)
float mnistInputData[MNIST_INPUT_SIZE];

// Hardcoded MNIST Data (populated in setup with a stylized '1')
// REPLACE WITH REAL MNIST DATA (784 float values 0.0-255.0) FOR ACCURACY TESTING!
float hardcodedMnistData[MNIST_INPUT_SIZE];

// Shape definition for the predict function
uint32_t mnistInputShape[] = {1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH};
uint32_t mnistShapeLen = sizeof(mnistInputShape) / sizeof(mnistInputShape[0]);

// Pointer where the library will write the result buffer address (float)
float* predictionResultPtr = nullptr;

// LED pin to use (Green on Nicla Vision)
#define PREDICTION_LED LEDG
// --- END PREDICTION CONSTANTS and VARIABLES ---

// --- GLOBAL VARIABLES ---
// These will be set in setup() after camera initialization
int currentFrameWidth = 0;
int currentFrameHeight = 0;
// Starts in TEST mode (hardcoded data)
bool useCameraMode = false;
// Frame buffer to capture image from camera
FrameBuffer fb;
// --- END GLOBAL VARIABLES ---


// --- HELPER FUNCTIONS ---

/**
 * @brief Blinks an LED for a specified number of times.
 * @param ledPin The LED pin to blink.
 * @param count The number of blinks (default: infinite).
 */
void blinkLED(int ledPin, uint32_t count = 0xFFFFFFFF) {
  pinMode(ledPin, OUTPUT);
  while (count--) {
    digitalWrite(ledPin, LOW); // Turn LED ON
    delay(50);
    digitalWrite(ledPin, HIGH); // Turn LED OFF
    delay(50);
  }
}

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// ===    PREPROCESSIMAGE FUNCTION - VERSION WITH BOUNDING BOX DETECTION ===
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
/**
 * @brief Preprocesses the source image to fit MNIST input.
 * 1. Thresholding to find digit pixels (assuming dark digit on light background).
 * 2. Finds the Bounding Box of the digit in the source image.
 * 3. Resizes ONLY the bounding box content to 28x28 using AREA AVERAGING.
 * 4. INVERTS values (light->dark) to match MNIST (0=black, 255=white).
 * @param srcBuffer Pointer to the source image buffer (grayscale).
 * @param srcWidth Source image width (e.g., 160).
 * @param srcHeight Source image height (e.g., 120).
 * @param dstBuffer Pointer to the destination buffer (float, 28x28).
 * @param dstWidth Destination image width (MNIST_WIDTH).
 * @param dstHeight Destination image height (MNIST_HEIGHT).
 */
void preprocessImage(byte* srcBuffer, int srcWidth, int srcHeight, float* dstBuffer, int dstWidth, int dstHeight) {

    // --- 1. Thresholding and Find Bounding Box ---
    byte thresholdValue = 100; // THRESHOLD: Pixels BELOW this value are considered part of the digit. ADJUST IF NEEDED!
    int minX = srcWidth, minY = srcHeight, maxX = -1, maxY = -1;
    bool digitPixelFound = false;

    for (int y = 0; y < srcHeight; ++y) {
        for (int x = 0; x < srcWidth; ++x) {
            byte pixel = srcBuffer[y * srcWidth + x];
            if (pixel < thresholdValue) { // Dark pixel (potential digit)
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                digitPixelFound = true;
            }
        }
    }

    // Clear destination buffer (default black background)
    for(int i=0; i < dstWidth * dstHeight; ++i) {
        dstBuffer[i] = 0.0f; // MNIST black background
    }

    // If no dark pixels were found (no digit?), exit with black buffer
    if (!digitPixelFound || minX > maxX || minY > maxY) {
         // Serial.println("WARN: No digit found during preprocessing (or invalid BBox)."); // Debug
         return;
    }

    // Add a small margin to the bounding box to avoid cutting edges
    const int padding = 4; // Increase/decrease if needed
    minX = max(0, minX - padding);
    minY = max(0, minY - padding);
    maxX = min(srcWidth - 1, maxX + padding);
    maxY = min(srcHeight - 1, maxY + padding);

    // Calculate source bounding box dimensions
    int bbWidth = maxX - minX + 1;
    int bbHeight = maxY - minY + 1;

    // If bounding box is too small, it might be noise, exit.
    if (bbWidth < 5 || bbHeight < 5) {
        // Serial.println("WARN: Bounding Box too small, likely noise."); // Debug
        return;
    }

    // --- 2. Resize Bounding Box content to 28x28 with Area Averaging ---
    // Now, map destination pixels (28x28) to SOURCE pixels
    // but only within the bounding box (minX, minY) -> (maxX, maxY).
    float scaleX = (float)bbWidth / dstWidth;
    float scaleY = (float)bbHeight / dstHeight;

    for (int y = 0; y < dstHeight; ++y) { // Iterate over destination 28x28
        for (int x = 0; x < dstWidth; ++x) {

            // Calculate the corresponding source area *within the bounding box*
            float srcStartX_f = minX + x * scaleX;
            float srcStartY_f = minY + y * scaleY;
            float srcEndX_f = minX + (x + 1) * scaleX;
            float srcEndY_f = minY + (y + 1) * scaleY;

            int srcStartX = (int)srcStartX_f;
            int srcStartY = (int)srcStartY_f;
            int srcEndX = (int)srcEndX_f; if(srcEndX_f > srcEndX) srcEndX++; // Implicit ceiling for upper bound
            int srcEndY = (int)srcEndY_f; if(srcEndY_f > srcEndY) srcEndY++; // Implicit ceiling for upper bound

            // Constrain to actual bounding box limits (ensures valid indices)
            // Use maxX+1 / maxY+1 for upper limits because loop is < end
            srcStartX = constrain(srcStartX, minX, maxX + 1);
            srcStartY = constrain(srcStartY, minY, maxY + 1);
            srcEndX = constrain(srcEndX, srcStartX, maxX + 1);
            srcEndY = constrain(srcEndY, srcStartY, maxY + 1);

            // Calculate average in the source area INSIDE the bounding box
            float sum = 0;
            int count = 0;
            for (int sy = srcStartY; sy < srcEndY; ++sy) {
                // y-index check (should be superfluous)
                if (sy < 0 || sy >= srcHeight) continue;
                for (int sx = srcStartX; sx < srcEndX; ++sx) {
                     // x-index check (should be superfluous)
                     if (sx < 0 || sx >= srcWidth) continue;
                     sum += srcBuffer[sy * srcWidth + sx]; // Access source pixel
                     count++;
                }
            }
            // Use average value, or consider light background if count is 0
            float avgPixelValue = (count > 0) ? (sum / count) : 255.0f;

            // --- 3. Invert for MNIST format and save as float ---
            // MNIST expects black background (0) and white digit (255)
            dstBuffer[y * dstWidth + x] = 255.0f - avgPixelValue;

             // --- OPTIONAL: Thresholding (after inversion) ---
            /*
            const float threshold = 128.0f; // THRESHOLD (value between 0 and 255 on INVERTED value)
            if (dstBuffer[y * dstWidth + x] > threshold) { // Above threshold -> White
                 dstBuffer[y * dstWidth + x] = 255.0f;
            } else { // Below threshold -> Black
                 dstBuffer[y * dstWidth + x] = 0.0f;
            }
            */
           // --- END OPTIONAL THRESHOLDING ---
        }
    }
     // Debug: uncomment to see confirmation
     // Serial.println("INFO: Preprocessing with Bounding Box and Area Averaging completed.");
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// ===      END PREPROCESSIMAGE FUNCTION - BOUNDING BOX          ===
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


/**
 * @brief Finds the index of the class with the highest score in the results array.
 * @param scores Pointer to the scores array (float).
 * @param numClasses Number of classes (and size of the scores array).
 * @return The index (int) of the class with the highest score (0-9), or -1 on error.
 */
int findPredictedDigit(float* scores, int numClasses) {
    // Validity check
    if (scores == nullptr || numClasses <= 0) {
        return -1; // Return -1 to indicate an error
    }
    int maxIndex = 0; // Index of the maximum score found so far
    float maxScore = scores[0]; // Maximum score found so far
    // Iterate to find the maximum
    for (int i = 1; i < numClasses; ++i) {
        if (scores[i] > maxScore) {
            maxScore = scores[i]; // Update maximum score
            maxIndex = i; // Update index of maximum
        }
    }
    return maxIndex; // Return the index of the class with the highest score
}

/**
 * @brief Populates the hardcodedMnistData array with a simple test pattern
 * resembling a digit '1'.
 * REPLACE this function or its content to use real MNIST data for accuracy tests.
 */
void populateHardcodedData() {
  Serial.println("INFO: Populating hardcoded data with test pattern ('1')...");
  // 1. Clear array (black background = 0.0f)
  for (int i = 0; i < MNIST_INPUT_SIZE; ++i) { hardcodedMnistData[i] = 0.0f; }
  // 2. Draw main vertical line (slightly right of center)
  int mainStrokeX = 16; for (int y = 5; y <= 23; ++y) { int index = y * MNIST_WIDTH + mainStrokeX; if (index >= 0 && index < MNIST_INPUT_SIZE) hardcodedMnistData[index] = 255.0f; if (mainStrokeX > 0) { int index_left = y * MNIST_WIDTH + (mainStrokeX - 1); if (index_left >= 0 && index_left < MNIST_INPUT_SIZE) hardcodedMnistData[index_left] = 150.0f; } if (mainStrokeX < MNIST_WIDTH - 1 ) { int index_right = y * MNIST_WIDTH + (mainStrokeX + 1); if (index_right >= 0 && index_right < MNIST_INPUT_SIZE) hardcodedMnistData[index_right] = 80.0f; } }
  // 3. Add a small serif at the top left
  int serif1_idx = 5 * MNIST_WIDTH + (mainStrokeX - 2); int serif2_idx = 6 * MNIST_WIDTH + (mainStrokeX - 3); int serif3_idx = 7 * MNIST_WIDTH + (mainStrokeX - 2); if (serif1_idx >=0 && serif1_idx < MNIST_INPUT_SIZE) hardcodedMnistData[serif1_idx] = 200.0f; if (serif2_idx >=0 && serif2_idx < MNIST_INPUT_SIZE) hardcodedMnistData[serif2_idx] = 180.0f; if (serif3_idx >=0 && serif3_idx < MNIST_INPUT_SIZE) hardcodedMnistData[serif3_idx] = 150.0f;
  // 4. Add a small base at the bottom
  int baseY = 24; for (int x = mainStrokeX - 3; x <= mainStrokeX + 1; ++x) { int index = baseY * MNIST_WIDTH + x; if (x >= 0 && x < MNIST_WIDTH && index >= 0 && index < MNIST_INPUT_SIZE) { hardcodedMnistData[index] = 200.0f; } } int base_attach_idx = (baseY-1) * MNIST_WIDTH + mainStrokeX; if (base_attach_idx >=0 && base_attach_idx < MNIST_INPUT_SIZE) hardcodedMnistData[base_attach_idx] = 255.0f;
  Serial.println("INFO: Hardcoded data ('1') populated."); Serial.println("      NOTE: This is a stylized '1', NOT from a real dataset.");
}
// --- END HELPER FUNCTIONS ---


// =========================================================================
// ===                         SETUP                                     ===
// =========================================================================
void setup() {
  // Initialize LED pins as OUTPUT
  pinMode(LED_BUILTIN, OUTPUT); // Built-in RGB LED
  pinMode(LEDR, OUTPUT);        // Red component of RGB LED
  pinMode(LEDG, OUTPUT);        // Green component of RGB LED
  pinMode(LEDB, OUTPUT);        // Blue component of RGB LED
  pinMode(PREDICTION_LED, OUTPUT); // Green LED used for prediction 1-5 indication

  // Turn off all LEDs on startup (HIGH state turns off LEDs on Nicla)
  digitalWrite(LED_BUILTIN, HIGH);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  digitalWrite(PREDICTION_LED, HIGH); // Ensure prediction LED is off

  // Initialize Serial communication
  Serial.begin(115200);
  // You might want to wait for serial connection for initial debug:
  // while (!Serial);
  Serial.println("\n================================================");
  Serial.println("MNIST Predict Sketch - Test Mode / Camera Mode");
  Serial.println("Target: Arduino Nicla Vision");
  Serial.println("================================================");
  Serial.println("Serial Commands:");
  Serial.println(" 'c' -> Activate Camera Mode");
  Serial.println(" 't' -> Activate Test Mode (Hardcoded Data)");
  Serial.println("------------------------------------------------");

  // Populate EXAMPLE hardcoded data (now draws a '1')
  populateHardcodedData();

  // Configure the Camera
  // Set the desired camera resolution.
  // CAMERA_R160x120 is a common and supported resolution.
  // If unavailable, try CAMERA_QVGA (320x240) or others defined by the library.
  uint8_t cameraResolution = CAMERA_R160x120;
  Serial.print("Configuring Camera...");
  Serial.print(" Resolution: ");

  // Set global width and height variables based on the chosen constant
  // This avoids calling non-existent .width()/.height() methods later.
  if (cameraResolution == CAMERA_R160x120) {
      currentFrameWidth = 160;
      currentFrameHeight = 120;
      Serial.print("160x120");
  } else if (cameraResolution == CAMERA_R320x240) { // Example if using QVGA
      currentFrameWidth = 320;
      currentFrameHeight = 240;
      Serial.print("320x240");
  }
  // Add other supported resolutions here if needed
  else {
      // Resolution not handled or constant incorrect
      currentFrameWidth = 0;
      currentFrameHeight = 0;
      Serial.print("NOT SUPPORTED!");
  }
  Serial.print(", Mode: ");
  Serial.println(IMAGE_MODE == CAMERA_GRAYSCALE ? "Grayscale" : "Other"); // Print mode

  // Check if dimensions were set correctly
  if (currentFrameWidth == 0 || currentFrameHeight == 0) {
      Serial.println("\nCRITICAL ERROR: Frame dimensions not set for chosen resolution!");
      Serial.println("Check 'cameraResolution' constant and the if/else if block.");
      while(1) { blinkLED(LEDR, 1); delay(200); } // Total lock with red LED
  }

  // Initialize the camera with chosen resolution, mode, and desired FPS
  Serial.print("Initializing Camera HW...");
  if (!cam.begin(cameraResolution, IMAGE_MODE, 30)) { // Try 30 FPS
     Serial.println(" FAILED!");
     Serial.println("CRITICAL ERROR: Failed to initialize camera.");
     while(1) { blinkLED(LEDR, 1); delay(100); } // Total lock with red LED
  } else {
     Serial.println(" OK.");
  }

  // Confirmation message for the data format expected by the library
  Serial.println("------------------------------------------------");
  Serial.println("INFO: Ensure lib_zant.h declares");
  Serial.println("      predict() with 'const float* input'.");
  Serial.println("      Sent data will be float 0.0f-255.0f.");
  Serial.println("------------------------------------------------");

  Serial.println("\nSetup completed.");
  if (useCameraMode) { Serial.println("Initial Mode: CAMERA"); }
  else { Serial.println("Initial Mode: TEST (Hardcoded '1' Data)"); }
  Serial.println("Starting loop...\n");
  blinkLED(LED_BUILTIN, 3); // Blink RGB LED 3 times = Setup OK
}


// =========================================================================
// ===                          LOOP                                     ===
// =========================================================================
void loop() {

  // 1. Check Serial Input for Mode Change
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'c' || command == 'C') {
      if (!useCameraMode) { useCameraMode = true; Serial.println("\n---> SWITCHING TO CAMERA MODE <---\n"); }
    } else if (command == 't' || command == 'T') {
       if (useCameraMode) { useCameraMode = false; Serial.println("\n---> SWITCHING TO TEST MODE (Hardcoded '1' Data) <---\n"); }
    }
    while(Serial.available() > 0) { Serial.read(); } // Clear buffer
  }

  // Pointer to the input data to use for the current prediction
  const float* currentInputData = nullptr;
  bool dataReady = false;

  // 2. Get Input Data based on mode
  if (useCameraMode) {
    // --- Camera Mode ---
    if (cam.grabFrame(fb, 3000) == 0) { // 3 second timeout
      byte* cameraBuffer = fb.getBuffer(); // Get pointer to camera buffer
      // Check if frame dimensions are valid (set in setup)
      if (currentFrameWidth > 0 && currentFrameHeight > 0) {
        // Perform preprocessing (now with Bounding Box and Area Averaging)
        preprocessImage(cameraBuffer, currentFrameWidth, currentFrameHeight,
                        mnistInputData, MNIST_WIDTH, MNIST_HEIGHT);

        // <<<<< START DEBUG VISUALIZATION BLOCK (Optional) >>>>>
        #ifdef DEBUG_PREPROCESSED_IMAGE
          Serial.println("--- Preprocessed Data (Camera Mode - 28x28 BBox Sample) ---");
          for (int y = 0; y < MNIST_HEIGHT; y+=2) { // Print every other row
            for (int x = 0; x < MNIST_WIDTH; x+=2) { // Print every other column
               int idx = y * MNIST_WIDTH + x;
               int val = (int)mnistInputData[idx]; // Show approximate integer value
               if (val < 50) Serial.print(".");      // Almost black
               else if (val < 150) Serial.print("o"); // Gray
               else Serial.print("#");      // Almost white
            }
            Serial.println();
          }
          Serial.println("-------------------------------------------------------");
        #endif
        // <<<<< END DEBUG VISUALIZATION BLOCK >>>>>

        // Set data pointer for prediction
        currentInputData = mnistInputData;
        dataReady = true;

      } else { Serial.println("CAMERA ERROR: Invalid frame dimensions!"); }
    } else { Serial.println("CAMERA ERROR: Frame capture failed!"); for(int i=0; i<5; i++) { blinkLED(LEDR, 1); delay(50); } }
  } else {
    // --- Test Mode (Hardcoded Data) ---
    currentInputData = hardcodedMnistData; // Use the hardcoded data
    dataReady = true;
    delay(100); // Small delay in test mode to avoid flooding serial too fast
  }

  // 3. Perform Prediction (only if data is ready and valid)
  if (dataReady && currentInputData != nullptr) {

    // ---> START TIME MEASUREMENT <---
    unsigned long startTime = micros(); // Get time before the call

    // Perform inference/prediction
    predict(
        currentInputData,    // Pointer to selected data (camera or hardcoded)
        mnistInputShape,     // Pointer to shape {1, 1, 28, 28}
        mnistShapeLen,       // Length of shape (4)
        &predictionResultPtr // Address of pointer for the result (float*)
    );

    unsigned long endTime = micros(); // Get time after the call
    // ---> END TIME MEASUREMENT <---


    // 4. Analyze Result
    if (predictionResultPtr != nullptr) {
      // Calculate duration (handles micros() rollover if endTime < startTime)
      unsigned long duration = endTime - startTime;

      // Find the index (digit 0-9) with the highest score
      int predictedDigit = findPredictedDigit(predictionResultPtr, MNIST_CLASSES);

      // Print results
      Serial.print("===> ");
      if (!useCameraMode) Serial.print("[TEST MODE '1'] ");
      Serial.print("Predicted Class: ");
      Serial.print(predictedDigit);

      // ---> Print prediction time <---
      Serial.print(" (Predict Time: ");
      Serial.print(duration);
      Serial.println(" us)"); // us = microseconds
      // ---------------------------------------

      Serial.flush(); // Ensure serial output is sent

      // 5. Control LED based on prediction
      if (predictedDigit >= 1 && predictedDigit <= 5) {
          digitalWrite(PREDICTION_LED, LOW); // Turn Green LED ON if digit is 1-5
      } else {
          digitalWrite(PREDICTION_LED, HIGH); // Turn Green LED OFF otherwise
      }

    } else {
      // Error: predict() returned a null pointer
      Serial.println("ERROR: predict() returned a null pointer!");
      digitalWrite(PREDICTION_LED, HIGH); // Ensure LED is off
    }
  } else if (useCameraMode) {
    // If we are in camera mode but data is not ready (grabFrame or preprocess error)
    digitalWrite(PREDICTION_LED, HIGH); // Ensure prediction LED is off
  }

  // Pause before next loop iteration
  delay(400); // Keep a reasonable delay to avoid overloading
}