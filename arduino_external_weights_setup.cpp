#include <Arduino.h>
#include <cstring>
// #include "zant/weights_io.h"  // Questo sarà incluso dalla libreria Zant
// #include "lib_mobilenet_v2.h" // Il tuo modello generato

// Dichiarazioni per le funzioni Zant (normalmente in headers)
extern "C" {
    void zant_init_weights_io(void);
    void zant_register_weight_callback(int (*callback)(size_t, uint8_t*, size_t));
    int predict(float* input, uint32_t* input_shape, uint32_t shape_len, float** result);
}

// Indirizzo base della memoria esterna dove sono caricati i pesi
#define EXTERNAL_FLASH_BASE 0x90000000

// Callback per leggere i pesi dalla memoria esterna
int external_flash_read(size_t offset, uint8_t* buffer, size_t size) {
    // Calcola l'indirizzo fisico nella memoria esterna
    const uint8_t* source = (const uint8_t*)(EXTERNAL_FLASH_BASE + offset);
    
    // Copia i dati (la memoria esterna è mappata direttamente)
    memcpy(buffer, source, size);
    
    return 0; // Successo
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("🚀 Inizializzazione Zant AI con pesi esterni...");
    
    // Configura il sistema di lettura pesi
    zant_init_weights_io();
    zant_register_weight_callback(external_flash_read);
    
    Serial.println("✅ Sistema configurato!");
    Serial.println("📍 Pesi caricati da memoria esterna @ 0x90000000");
}

void loop() {
    // Prepara input di test (32x32x3 = 3072 elementi)
    static float input_data[3072];
    static uint32_t input_shape[4] = {1, 3, 32, 32};
    static float* output_ptr;
    
    // Riempi con dati di test (sostituisci con dati reali)
    for (int i = 0; i < 3072; i++) {
        input_data[i] = (float)(i % 256) / 255.0f;
    }
    
    Serial.println("🔄 Eseguendo predizione...");
    
    // Esegui predizione
    int result = predict(input_data, input_shape, 4, &output_ptr);
    
    if (result == 0) {
        Serial.println("✅ Predizione completata!");
        Serial.print("Risultati: ");
        for (int i = 0; i < 4; i++) {
            Serial.print(output_ptr[i]);
            if (i < 3) Serial.print(", ");
        }
        Serial.println();
    } else {
        Serial.print("❌ Errore predizione: ");
        Serial.println(result);
    }
    
    delay(5000);
}

/*
ISTRUZIONI PER L'USO:

1. Compila il modello modificato (con caricamento dinamico)
2. Carica i pesi in memoria esterna:
   dfu-util -a 1 -D mobilenet_weights.bin -s 0x90000000

3. Compila e carica questo sketch
4. Il modello userà automaticamente i pesi dalla memoria esterna

VANTAGGI:
- Modello principale molto più piccolo (fit in flash interna)  
- Pesi in memoria esterna (4MB+ disponibili)
- Caricamento on-demand (solo i pesi necessari)
- Performance mantenute (memoria mappata)
*/ 