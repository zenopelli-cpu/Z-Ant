# MNIST Digit Recognition Example for Arduino Nicla Vision

This example demonstrates how to use a pre-compiled static library (`libZant.a`) to perform MNIST digit recognition on an Arduino Nicla Vision board. The sketch captures images from the camera, preprocesses them, and uses the static library's `predict` function to classify the digit (0-9).

## Requirements

### Hardware
* Arduino Nicla Vision

### Software
* Arduino IDE (1.8.19 or later) or Arduino CLI
* Arduino Mbed OS Nicla Boards core (Version 4.2.4 or compatible, installed via Board Manager)

### Libraries
1.  **Zant Static Library:** The custom pre-compiled library (`libZant.a` and header `lib_zant.h`). You need to install this manually (see Installation steps).
2.  **Camera Libraries:** `Camera`, `GC2145`, `Wire`. These are typically bundled with the Arduino Mbed OS Nicla Boards core and should be installed automatically with it.

## Installation

1.  **Install the Arduino Core:** If you haven't already, install the "Arduino Mbed OS Nicla Boards" core using the Arduino IDE's Board Manager or via Arduino CLI.
2.  **Install the Custom Static Library (`Zant`):**
    * Locate your Arduino sketchbook folder. You can find the path in the Arduino IDE under `File > Preferences > Sketchbook location`.
    * Inside the sketchbook folder, find or create the `libraries` subfolder.
    * Copy the entire `Zant` library folder (the one containing `library.properties`, the `src` folder with `lib_zant.h`, and `src/cortex-m7/libZant.a`) into this `libraries` folder.
    * The final path should look something like: `<Sketchbook>/libraries/Zant/`
3.  **Restart Arduino IDE:** If the IDE was open, close and reopen it to ensure it detects the newly added library.

## Running the Example

1.  **Open the Sketch:** Open this example sketch file (`YourSketchName.ino`) in the Arduino IDE.
2.  **Select Board:** Go to `Tools > Board > Arduino Mbed OS Nicla Boards` and select "Nicla Vision".
3.  **Select Port:** Go to `Tools > Port` and select the correct serial port for your connected Nicla Vision.
4.  **Upload:** Click the "Upload" button (or use `arduino-cli upload ...`).
5.  **Open Serial Monitor:** Go to `Tools > Serial Monitor` or use the Serial Monitor icon. Set the baud rate to **115200**.

## Sketch Functionality

The sketch operates in two modes, switchable via the Serial Monitor:

1.  **Test Mode (Default):**
    * Starts automatically when the sketch runs.
    * Uses hardcoded image data representing a stylized digit '1'.
    * Calls the `predict` function with this hardcoded data.
    * Prints the predicted class (digit 0-9) and the prediction time (in microseconds) to the Serial Monitor. Example output:
        ```
        ===> [TEST MODE '1'] Predicted Class: 1 (Predict Time: 1500 us)
        ```
    * This mode is useful for verifying that the static library is correctly linked and the `predict` function executes without crashing.

2.  **Camera Mode:**
    * Activated by sending `c` (or `C`) through the Serial Monitor.
    * Continuously captures frames from the Nicla Vision camera (at 160x120 grayscale resolution).
    * **Preprocesses** the captured image:
        * Finds potential digit pixels using thresholding (assuming dark digit on light background).
        * Calculates the bounding box of the detected digit.
        * Resizes the content *within the bounding box* to 28x28 pixels using area averaging.
        * **Inverts** the pixel values (so the digit becomes white/high value, background becomes black/low value) to match the typical MNIST format.
        * Stores the result as a 28x28 array of `float` values ranging from 0.0f to 255.0f.
    * Calls the `predict` function with the preprocessed camera image data.
    * Prints the predicted class and prediction time. Example output:
        ```
        ===> Predicted Class: 3 (Predict Time: 1850 us)
        ```
    * **Controls the Green LED:** Turns the onboard Green LED (LEDG) ON (LOW state) if the predicted digit is between 1 and 5 (inclusive), otherwise turns it OFF (HIGH state).

3.  **Switching Back:** Send `t` (or `T`) via the Serial Monitor to return to Test Mode.

## Notes & Troubleshooting

* **Hardcoded Data:** The '1' used in Test Mode is stylized and drawn directly in the code. For *true* accuracy testing, replace the `populateHardcodedData` function's content with actual 784 pixel values (floats 0.0-255.0) from a real MNIST dataset image.
* **Preprocessing Tuning:** The image preprocessing (especially the `thresholdValue` and `padding` in the `preprocessImage` function) might need adjustment based on your lighting conditions and how you present the digits to the camera for optimal results in Camera Mode. Try using clear, high-contrast digits (e.g., dark marker on white paper).
* **Prediction Accuracy:** The accuracy in Camera Mode depends heavily on the quality of the preprocessing, lighting, centering of the digit, and the robustness of the underlying ML model in the static library.
* **Library Issues:** If you encounter `undefined reference to 'predict'` errors during linking, double-check the library installation path (`libraries/Zant`), the structure (`src/cortex-m7/libZant.a`), and the content of `library.properties` (`precompiled=true`). Ensure you restarted the IDE.
* **Debug Output:** You can uncomment `#define DEBUG_PREPROCESSED_IMAGE` at the top of the sketch to print a textual representation of the 28x28 preprocessed image to the Serial Monitor in Camera Mode, which can help diagnose preprocessing issues.