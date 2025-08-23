# Integrazione Zant AI su Arduino Nicla Vision

## 📋 Prerequisiti

- Arduino IDE 2.x o PlatformIO
- Arduino Nicla Vision board
- Board package Arduino Mbed OS Nano
- Libreria Arduino_BMI270_BMM150 (sensori)
- Libreria Camera (per cattura immagini)

## 🚀 Setup Progetto

### 1. Struttura Directory Arduino

```
Il_Tuo_Sketch/
├── Il_Tuo_Sketch.ino
├── ZantAI.h
├── ZantAI.cpp
├── libzant_nicla.a
└── weights_data.cpp         // File pesi generato
```

### 2. Copia File Necessari

Dalla directory `arduino/` di Zant, copia:
- `ZantAI.h` e `ZantAI.cpp` → nel tuo sketch
- `libzant_nicla.a` → nel tuo sketch
- `generated/mobilenet_v2/static_parameters.zig` → converti in weights_data.cpp

### 3. Configurazione Arduino IDE

#### platform.txt (per advanced users)
Aggiungi al `platform.txt` della board Nicla Vision:

```ini
# Zant AI Library Support
compiler.libraries.ldflags=-Wl,--whole-archive {build.path}/sketch/libzant_nicla.a -Wl,--no-whole-archive

# Linker flags for Zant
build.extra_flags=-DZANT_ARDUINO -DZANT_WEIGHTS_IO_MODE
```

#### Oppure modifica sketch direttamente:

Aggiungi al tuo `.ino`:
```cpp
// Linker directives per includere la libreria statica
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
asm(".section .zant_lib");
asm(".incbin \"libzant_nicla.a\"");
#pragma GCC diagnostic pop
```

### 4. Conversione Pesi in Array C

I pesi del modello devono essere convertiti da Zig a formato C. Crea `weights_data.cpp`:

```cpp
#include <stdint.h>

// Pesi del modello (estratti da static_parameters.zig)
extern "C" {
    // Array principale dei pesi - DA AGGIORNARE CON I TUOI DATI
    const float model_weights[] = {
        // Inserisci qui i pesi dal file static_parameters.zig
        // Esempio: 0.123f, -0.456f, 0.789f, ...
    };
    
    const size_t model_weights_size = sizeof(model_weights);
}
```

## 💻 Codice di Esempio

### Sketch Base

```cpp
#include "ZantAI.h"

ZantAI zant;
float predictions[4];

void setup() {
    Serial.begin(115200);
    
    // Inizializza Zant AI
    zant.begin();
    
    Serial.println("Zant AI ready!");
}

void loop() {
    // I tuoi dati di input (32x32x3 RGB normalizzati)
    float input_data[3072];
    
    // ... prepara input_data ...
    
    // Esegui predizione
    int result = zant.predictRaw(input_data, predictions);
    
    if (result == 0) {
        Serial.println("Prediction successful!");
        // Stampa risultati
        for (int i = 0; i < 4; i++) {
            Serial.print("Class ");
            Serial.print(i);
            Serial.print(": ");
            Serial.println(predictions[i]);
        }
    }
    
    delay(1000);
}
```

### Con Camera Nicla Vision

```cpp
#include "ZantAI.h"
#include "camera.h"

ZantAI zant;
uint8_t image_buffer[32*32*3];
float predictions[4];

void setup() {
    Serial.begin(115200);
    
    // Inizializza camera
    Camera.begin(CAMERA_R320x240, IMAGE_MODE_RGB888, 30);
    
    // Inizializza AI
    zant.begin();
}

void loop() {
    // Cattura immagine
    Camera.grab();
    uint8_t* camera_data = Camera.getBuffer();
    
    // Ridimensiona a 32x32 (implementa la tua funzione)
    resizeImage(camera_data, 320, 240, image_buffer, 32, 32);
    
    // Predici
    int result = zant.predictImage(image_buffer, 32, 32, predictions);
    
    if (result == 0) {
        // Mostra risultati
        for (int i = 0; i < 4; i++) {
            Serial.print("Class ");
            Serial.print(i);
            Serial.print(": ");
            Serial.print(predictions[i] * 100.0f);
            Serial.println("%");
        }
    }
    
    delay(3000);
}
```

## ⚙️ Configurazioni Avanzate

### Custom Weight Loading

Se vuoi caricare i pesi da SD card o altro storage:

```cpp
#include "SD.h"

int sdcard_weight_callback(size_t offset, uint8_t* buffer, size_t size) {
    File weights_file = SD.open("model_weights.bin");
    if (!weights_file) return -1;
    
    weights_file.seek(offset);
    size_t read_bytes = weights_file.read(buffer, size);
    weights_file.close();
    
    return (read_bytes == size) ? 0 : -1;
}

void setup() {
    // ... setup normale ...
    
    // Inizializza SD card
    SD.begin();
    
    // Imposta callback personalizzato
    zant.setWeightCallback(sdcard_weight_callback);
    zant.begin();
}
```

### Ottimizzazioni Memoria

Per Nicla Vision con memoria limitata:

1. **Usa quantizzazione**: Compila il modello con `-Dquantize=true`
2. **Buffer statici**: Evita allocazioni dinamiche
3. **Ridimensiona input**: Usa modelli più piccoli se possibile

```cpp
// Ottimizzazioni compiler
#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")

// Buffer statici per evitare heap fragmentation
static uint8_t static_image_buffer[32*32*3];
static float static_input_buffer[3072];
static float static_output_buffer[4];
```

## 🐛 Troubleshooting

### Errori Comuni

1. **"Library not found"**
   - Assicurati che `libzant_nicla.a` sia nella directory dello sketch
   - Verifica platform.txt o use pragma

2. **"Out of memory"**
   - Riduci dimensioni buffer
   - Usa static allocation invece di heap
   - Considera modelli più piccoli

3. **"Prediction returns -1"**
   - Verifica che l'input sia nel formato corretto (32x32x3)
   - Controlla che i pesi siano caricati correttamente
   - Debug con Serial output

### Debug

Attiva debug verbose:

```cpp
#define ZANT_DEBUG 1

void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    
    zant.begin();
}
```

## 📊 Performance

Su Arduino Nicla Vision (Cortex-M7 @ 480MHz):
- **MobileNet v2 (32x32)**: ~200-500ms per inference
- **Memoria RAM**: ~100-200KB durante inference
- **Memoria Flash**: ~1-5MB per i pesi del modello

## 🔗 Riferimenti

- [Arduino Nicla Vision Documentation](https://docs.arduino.cc/hardware/nicla-vision)
- [Arduino Camera Library](https://www.arduino.cc/reference/en/libraries/camera/)
- [Zant Documentation](../docs/WEIGHTS_IO_GUIDE.md) 