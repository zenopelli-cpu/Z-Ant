#ifndef MY_NICLA_LIB_PREDICT_H
#define MY_NICLA_LIB_PREDICT_H

// Include l'header standard per tipi interi a dimensione fissa come uint32_t
#include <stdint.h>

// Questo blocco assicura che il linkage C sia usato anche se l'header
// viene incluso in un file C++ (come lo sketch .ino di Arduino)
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Esegue l'inferenza del modello sulla base dei dati di input.
 *
 * @param input Puntatore al buffer contenente i dati di input (float).
 * Deve essere un array flat che rappresenta il tensore di input.
 * @param input_shape Puntatore a un array di uint32_t che descrive la
 * forma (dimensioni) del tensore di input (es. {1, 1, 28, 28}).
 * @param shape_len Numero di elementi nell'array input_shape (numero di dimensioni).
 * @param result Puntatore a un puntatore a float (float**). La funzione scriverà
 * l'indirizzo di memoria del buffer interno contenente il risultato
 * dell'inferenza nella locazione puntata da 'result'.
 * ATTENZIONE: Non deallocare la memoria puntata da *result,
 * appartiene alla libreria. Il contenuto è valido fino alla
 * successiva chiamata di predict().
 */
void predict(
    const float* input,
    const uint32_t* input_shape,
    uint32_t shape_len,
    float** result
);

// (Opzionale) Puoi dichiarare anche setLogFunction se prevedi di usarla
// typedef void (*log_callback_t)(const char*); // Definisci un tipo per il callback
// void setLogFunction(log_callback_t func);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // MY_NICLA_LIB_PREDICT_H
